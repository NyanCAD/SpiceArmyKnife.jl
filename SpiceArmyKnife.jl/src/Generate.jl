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
            target_dialects=[:ngspice]
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
            spice_dialect=:pspice,  # MicroCap syntax is closest to PSpice
        ),

        ArchiveConfig(
            "https://github.com/CedarEDA/Sky130PDK.jl/archive/refs/heads/main.zip",
            ["Sky130PDK.jl-main/sky130A/libs.tech/ngspice/sky130.lib.spice"],
            ["Sky130"],
            :lib; # FOSSi but large
            lib_section="tt",  # Only process "tt" (typical) corner to avoid duplicates
            device_blacklist=r"__parasitic|__base"i  # Skip parasitic and base implementation devices
        ),
        
        ArchiveConfig(
            "https://github.com/CedarEDA/GF180MCUPDK.jl/archive/refs/heads/main.zip",
            ["GF180MCUPDK.jl-main/model/sm141064.ngspice"],
            ["GF180MCU"],
            :lib; # FOSSi but large
            lib_section="typical"
        )
    ]
    
    # Process each archive
    for (i, config) in enumerate(archives)
        println("\nProcessing archive $(i)/$(length(archives)): $(config.url)")
        println("Category: $(join(config.base_category, " / "))")
        println("Mode: $(config.mode)")
        println("Entrypoints: $(config.entrypoints === nothing ? "auto-discover" : join(config.entrypoints, ", "))")
        println("Lib section: $(config.lib_section === nothing ? "all" : config.lib_section)")
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
        # Preserve database permissions and views before upload
        println("Preserving database permissions and views...")
        security_doc = fetch_security(url, headers)
        design_docs = fetch_design_docs(url, headers)

        if security_doc !== nothing
            println("  ✓ Saved database permissions")
        end
        if !isempty(design_docs)
            println("  ✓ Saved $(length(design_docs)) design document(s) (views)")
        end

        # Get existing revisions
        response = HTTP.get("$url/models/_all_docs", headers)
        existing_revs = Dict{String, String}()

        if response.status == 200
            data = JSON.parse(String(response.body))
            for row in data["rows"]
                existing_revs[row["id"]] = row["value"]["rev"]
            end
            println("Found $(length(existing_revs)) existing documents")
        end
        
        # Add revisions to models
        docs_with_revs = []
        for model in models
            doc = deepcopy(model)
            if haskey(existing_revs, doc["_id"])
                doc["_rev"] = existing_revs[doc["_id"]]
            end
            push!(docs_with_revs, doc)
        end
        
        # Upload in chunks
        total_uploaded = 0
        chunks = [docs_with_revs[i:min(i+chunk_size-1, end)] for i in 1:chunk_size:length(docs_with_revs)]
        
        for (i, chunk) in enumerate(chunks)
            println("Uploading chunk $i/$(length(chunks)) ($(length(chunk)) docs)...")
            
            response = HTTP.post("$url/models/_bulk_docs", headers, JSON.json(Dict("docs" => chunk)))
            
            if response.status in [200, 201]
                total_uploaded += length(chunk)
                println("  ✓ Chunk $i uploaded successfully")
            else
                println("  ✗ Chunk $i failed: $(response.status)")
                return false
            end
        end
        
        println("✓ Successfully uploaded $total_uploaded/$(length(models)) models to CouchDB!")

        # Restore permissions and views
        println("\nRestoring database permissions and views...")
        restore_security(url, headers, security_doc)
        restore_design_docs(url, headers, design_docs)

        return true
        
    catch e
        print("✗ Error uploading: ")
        showerror(stdout, e)
        println()
        return false
    end
end

"""
Fetch the database security document (permissions).
Returns nothing if the document doesn't exist or on error.
"""
function fetch_security(url, headers)
    try
        response = HTTP.get("$url/models/_security", headers)
        if response.status == 200
            return JSON.parse(String(response.body))
        end
    catch e
        # Security document might not exist in new databases
        if isa(e, HTTP.ExceptionRequest.StatusError) && e.status == 404
            return nothing
        end
        println("  ⚠ Warning: Could not fetch security document: $e")
    end
    return nothing
end

"""
Restore the database security document (permissions).
"""
function restore_security(url, headers, security_doc)
    if security_doc === nothing
        return true
    end

    try
        response = HTTP.put("$url/models/_security", headers, JSON.json(security_doc))
        if response.status in [200, 201]
            println("  ✓ Restored database permissions")
            return true
        else
            println("  ✗ Failed to restore permissions: $(response.status)")
            return false
        end
    catch e
        println("  ✗ Error restoring permissions: $e")
        return false
    end
end

"""
Fetch all design documents (views, shows, lists, etc.).
Returns a vector of design documents.
"""
function fetch_design_docs(url, headers)
    design_docs = []
    try
        # Fetch design documents using _all_docs with startkey and endkey
        query_params = "startkey=\"_design/\"&endkey=\"_design0\"&include_docs=true"
        response = HTTP.get("$url/models/_all_docs?$query_params", headers)

        if response.status == 200
            data = JSON.parse(String(response.body))
            for row in data["rows"]
                if haskey(row, "doc")
                    push!(design_docs, row["doc"])
                end
            end
        end
    catch e
        println("  ⚠ Warning: Could not fetch design documents: $e")
    end
    return design_docs
end

"""
Restore design documents (views) to the database.
"""
function restore_design_docs(url, headers, design_docs)
    if isempty(design_docs)
        return true
    end

    println("  Restoring $(length(design_docs)) design document(s)...")
    success = true

    for doc in design_docs
        try
            doc_id = doc["_id"]
            # Remove _rev to get the latest revision from the server
            delete!(doc, "_rev")

            # Try to get current revision
            try
                get_response = HTTP.get("$url/models/$doc_id", headers)
                if get_response.status == 200
                    current_doc = JSON.parse(String(get_response.body))
                    doc["_rev"] = current_doc["_rev"]
                end
            catch e
                # Document doesn't exist, that's ok
            end

            response = HTTP.put("$url/models/$doc_id", headers, JSON.json(doc))
            if response.status in [200, 201]
                println("    ✓ Restored design document: $doc_id")
            else
                println("    ✗ Failed to restore design document $doc_id: $(response.status)")
                success = false
            end
        catch e
            println("    ✗ Error restoring design document: $e")
            success = false
        end
    end

    return success
end

end # module Generate