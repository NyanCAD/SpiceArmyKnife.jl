"""
Model Database Generator

This module processes multiple SPICE model archives from various sources
and generates a unified JSON database for CouchDB upload.
"""
module Generate

using ..SpiceArmyKnife
using JSON
using HTTP
using StringEncodings

function (@main)(ARGS)
    println("=" ^ 80)
    println("SPICE Model Database Generator")
    println("=" ^ 80)
    
    # Combined results will be accumulated here
    all_models = Vector{Dict{String, Any}}()

    # Aggregate error statistics across all archives
    total_error_stats = Dict{String, Int}()
    total_failed_files = String[]
    total_files_processed = 0
    
    # Archive configurations
    archives = [
        ArchiveConfig(
            "https://ngspice.sourceforge.io/model-parameters/basic_models.7z",
            nothing,
            ["Basic"],
            :inline # public domain models
        ),

        ArchiveConfig(
            "https://www.cordellaudio.com/book/Cordell-Models.txt",
            nothing,
            ["Cordell"],
            :inline; # copyright Cordell Audio (preserved in model comments)
            target_simulators=[Ngspice()]
        ),

        ArchiveConfig(
            "https://ngspice.sourceforge.io/model-parameters/MicroCap-LIBRARY.7z",
            nothing,
            ["MicroCap"],
            :inline; # commercial restrictions may apply to some models
            file_device_types=Dict(
                "KemetCeramicCaps.LIB" => "capacitor",
                "KemetPolymerCaps.LIB" => "capacitor",
                "KemetTantalumCaps.LIB" => "capacitor",
                "Littelfuse_SIDACtor.lib" => "diode", # SIDACtor protection devices (thyristor-like)
                "vishayinductor.lib" => "inductor",
                "tdkbeads.lib" => "inductor", # ferrite beads
                "rectifie.lib" => "diode",
                "dei.lib" => "nmos", # power MOSFETs
                "m_rfdev.lib" => "npn", # RF transistors
                "nichicon.LIB" => "capacitor",
                "skyworks.lib" => "diode", # varactor tuning diodes
                "osram.lib" => "diode", # infrared LEDs
                "ntc.lib" => "resistor" # thermistors (temperature-dependent resistors)
            ),
            encoding=enc"ISO-8859-1",  # MicroCap files contain degree symbols in Latin-1 encoding
            simulator=Pspice(),  # MicroCap syntax is closest to PSpice
        ),

        ArchiveConfig(
            "https://nyancad.com/gh/fossi-foundation/skywater-pdk-libs-sky130_fd_pr/archive/refs/heads/main.zip",
            ["skywater-pdk-libs-sky130_fd_pr-main/combined_models/sky130.lib.spice"],
            ["Sky130"],
            :lib; # FOSSi but large
            lib_sections=["tt"],  # Only process "tt" (typical) corner to avoid duplicates
            device_blacklist=r"__parasitic|__base"i  # Skip parasitic and base implementation devices
        ),

        ArchiveConfig(
            "https://nyancad.com/gh/fossi-foundation/globalfoundries-pdk-libs-gf180mcu_fd_pr/archive/refs/heads/main.zip",
            ["globalfoundries-pdk-libs-gf180mcu_fd_pr-main/models/ngspice/sm141064.ngspice"],
            ["GF180MCU"],
            :lib; # FOSSi but large
            lib_sections=["typical"]
        ),

        ArchiveConfig(
            "https://nyancad.com/gh/IHP-GmbH/IHP-Open-PDK/archive/refs/heads/main.zip",
            [
                "IHP-Open-PDK-main/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib",
                "IHP-Open-PDK-main/ihp-sg13g2/libs.tech/ngspice/models/cornerMOShv.lib",
                "IHP-Open-PDK-main/ihp-sg13g2/libs.tech/ngspice/models/cornerCAP.lib",
                "IHP-Open-PDK-main/ihp-sg13g2/libs.tech/ngspice/models/cornerRES.lib",
                "IHP-Open-PDK-main/ihp-sg13g2/libs.tech/ngspice/models/cornerHBT.lib",
            ],
            ["IHP-SG13G2"],
            :lib; # FOSSi but large
            lib_sections=["mos_tt", "cap_typ", "res_typ", "hbt_typ"]
        )
    ]
    
    # Process each archive
    for (i, config) in enumerate(archives)
        println("\nProcessing archive $(i)/$(length(archives)): $(config.url)")
        println("Category: $(join(config.base_category, " / "))")
        println("Mode: $(config.mode)")
        println("Entrypoints: $(config.entrypoints === nothing ? "auto-discover" : join(config.entrypoints, ", "))")
        println("Lib sections: $(isempty(config.lib_sections) ? "all" : join(config.lib_sections, ", "))")
        println("Device blacklist: $(config.device_blacklist === nothing ? "none" : config.device_blacklist)")
        println("-" ^ 60)
        
        try
            result = process_archive(config)

            println("✓ Successfully processed $(result.models_generated) models")
            append!(all_models, result.mosaic_models)

            # Aggregate error statistics
            for (error_type, count) in result.error_stats
                total_error_stats[error_type] = get(total_error_stats, error_type, 0) + count
            end
            append!(total_failed_files, result.failed_files)
            total_files_processed += result.files_processed

        catch e
            print("✗ Error processing archive: ")
            showerror(stdout, e)
            println("\nContinuing with next archive...")
        end
    end
    
    # Generate final output
    output_file = "model_database.json"
    println("\n" * "=" ^ 80)
    println("GENERATION COMPLETE")
    println("=" ^ 80)
    println("Total models generated: $(length(all_models))")
    println("Total files processed: $total_files_processed")

    # Print aggregate error statistics
    if !isempty(total_error_stats)
        println("\nAggregate Error Statistics:")
        for (error_type, count) in sort(collect(total_error_stats), by=x->x[2], rev=true)
            println("  $error_type: $count occurrences")
        end
        println("\nTotal failed files: $(length(total_failed_files))")
        success_rate = round((total_files_processed - length(total_failed_files)) / total_files_processed * 100, digits=1)
        println("Success rate: $success_rate% ($(total_files_processed - length(total_failed_files))/$total_files_processed)")
    else
        println("\nNo parsing errors encountered!")
    end

    # Group by type for summary
    type_counts = Dict{String, Int}()
    for model in all_models
        model_type = model["type"]
        type_counts[model_type] = get(type_counts, model_type, 0) + 1
    end

    println("\nBreakdown by type:")
    for (type, count) in sort(collect(type_counts))
        println("  $type: $count models")
    end
    
    # Save to JSON file
    println("\nSaving to $output_file...")
    open(output_file, "w") do io
        JSON.print(io, all_models, 2)  # Pretty-print with 2-space indent
    end
    
    file_size_kb = round(stat(output_file).size / 1024, digits=1)
    println("✓ Saved $(length(all_models)) models to $output_file ($(file_size_kb) KB)")
    
    # Show sample models
    println("\nSample models:")
    for model in first(all_models, min(3, length(all_models)))
        println("  - $(model["name"]) ($(model["type"]))")
        println("    Category: $(join(model["category"], " / "))")
        println("    ID: $(model["_id"])")
    end
    
    # Upload to CouchDB
    upload_success = upload_to_couchdb(all_models)
    
    println("\n" * "=" ^ 80)
    if upload_success
        println("✓ GENERATION AND UPLOAD COMPLETE!")
    else
        println("✓ GENERATION COMPLETE - UPLOAD SKIPPED/FAILED")
        println("JSON file available at: $output_file")
    end
    println("=" ^ 80)
end

function upload_to_couchdb(models, chunk_size=100)
    """Upload models to CouchDB using bulk docs API in chunks."""
    url = rstrip(get(ENV, "COUCHDB_URL", "https://api.nyancad.com"), '/')
    user = get(ENV, "COUCHDB_ADMIN_USER", "admin")
    pass = get(ENV, "COUCHDB_ADMIN_PASS", "")
    
    if isempty(pass)
        println("⚠ No COUCHDB_ADMIN_PASS set - skipping CouchDB upload")
        return false
    end
    
    println("Uploading $(length(models)) models to CouchDB in chunks of $chunk_size...")
    
    headers = Dict(
        "Content-Type" => "application/json",
        "Authorization" => "Basic " * HTTP.base64encode("$user:$pass")
    )
    
    try
        # Delete existing database
        try
            HTTP.delete("$url/models", headers)
            println("Deleted existing 'models' database")
        catch
            println("Note: Database 'models' did not exist (will create fresh)")
        end

        # Create fresh database
        HTTP.put("$url/models", headers)
        println("Created fresh 'models' database")

        # Upload in chunks
        total_uploaded = 0
        chunks = [models[i:min(i+chunk_size-1, end)] for i in 1:chunk_size:length(models)]
        
        for (i, chunk) in enumerate(chunks)
            println("Uploading chunk $i/$(length(chunks)) ($(length(chunk)) docs)...")

            response = HTTP.post("$url/models/_bulk_docs", headers, JSON.json(Dict("docs" => chunk)))

            if response.status in [200, 201]
                # Parse response to check per-document results
                results = JSON.parse(String(response.body))

                successful = 0
                failed = 0
                for result in results
                    if haskey(result, "error")
                        failed += 1
                        error_msg = get(result, "reason", "Unknown error")
                        println("    ✗ Document '$(result["id"])' failed: $(result["error"]) - $error_msg")
                    else
                        successful += 1
                    end
                end

                total_uploaded += successful

                if failed > 0
                    println("  ⚠ Chunk $i: $successful succeeded, $failed failed")
                else
                    println("  ✓ Chunk $i uploaded successfully ($successful docs)")
                end
            else
                println("  ✗ Chunk $i failed with HTTP $(response.status)")
                return false
            end
        end
        
        total_failed = length(models) - total_uploaded
        if total_failed > 0
            println("⚠ Upload completed with partial success: $total_uploaded succeeded, $total_failed failed out of $(length(models)) models")
            return false  # Return false to indicate partial failure
        else
            println("✓ Successfully uploaded $total_uploaded/$(length(models)) models to CouchDB!")
            return true
        end
        
    catch e
        print("✗ Error uploading: ")
        showerror(stdout, e)
        println()
        return false
    end
end

end # module Generate