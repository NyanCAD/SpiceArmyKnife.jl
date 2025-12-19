"""
    ExtractionConfig

Configuration for extracting definitions from SPICE/Spectre AST.
"""
mutable struct ExtractionConfig
    parsed_files::Dict{String, Any}
    includepaths::Vector{String}
    libraries::Set{Tuple{String,String}}
    lib_sections::Vector{String}
    device_blacklist::Union{Regex, Nothing}
    current_file::String
    file_device_types::Dict{String, String}  # filename -> device type override

    function ExtractionConfig(; parsed_files=Dict{String, Any}(),
                            includepaths=String[], libraries=Set{Tuple{String,String}}(),
                            lib_sections=String[], device_blacklist=nothing, current_file="",
                            file_device_types=Dict{String, String}())
        new(parsed_files, includepaths, libraries, lib_sections, device_blacklist, current_file, file_device_types)
    end
end

# Helper function to create a new config for includes
function deeper(config::ExtractionConfig; new_includepaths=nothing, new_file=nothing)
    ExtractionConfig(
        parsed_files=config.parsed_files,
        includepaths=new_includepaths === nothing ? config.includepaths : new_includepaths,
        libraries=config.libraries,
        lib_sections=config.lib_sections, device_blacklist=config.device_blacklist,
        current_file=new_file === nothing ? config.current_file : new_file,
        file_device_types=config.file_device_types)
end

"""
    ArchiveConfig

Configuration for processing an archive with SPICE/Spectre models.

Fields:
- url: URL to download archive from
- entrypoints: Either Vector{String} of specific paths or nothing for auto-discovery
- base_category: Base category path for Mosaic format
- mode: Template mode (:inline, :include, :lib)
- lib_sections: Vector of .lib section names to whitelist. Only sections in this list will be processed. Empty vector processes all sections.
- device_blacklist: Regex pattern to skip device names (use alternation like r"__parasitic|__base"i)
- file_device_types: Dict mapping filename -> device type for file-level overrides
- encoding: File encoding (default: enc"UTF-8"). Use enc"ISO-8859-1" for older SPICE files with degree symbols
- simulator: Simulator for parsing (e.g., Ngspice(), Hspice()). Default: Ngspice()
- strict: Enforce dialect-specific parsing rules. Default: false
- target_simulators: Vector of simulators to generate converted templates for (e.g., [Ngspice()]). Default: AbstractSimulator[] (no conversions)
"""
struct ArchiveConfig
    url::String
    entrypoints::Union{Vector{String}, Nothing}
    base_category::Vector{String}
    mode::Symbol
    lib_sections::Vector{String}
    device_blacklist::Union{Regex, Nothing}
    file_device_types::Dict{String, String}
    encoding::Encoding
    simulator::AbstractSimulator
    strict::Bool
    target_simulators::Vector{AbstractSimulator}

    function ArchiveConfig(url, entrypoints, base_category, mode; lib_sections=String[], device_blacklist=nothing, file_device_types=Dict{String, String}(), encoding=enc"UTF-8", simulator=Ngspice(), strict=false, target_simulators=AbstractSimulator[])
        new(url, entrypoints, base_category, mode, lib_sections, device_blacklist, file_device_types, encoding, simulator, strict, target_simulators)
    end
end

"""
    generate_template_code(code, mode, archive_url, file_path)

Generate template code for Mosaic format.
- mode = :inline: returns the actual code
- mode = :include: returns .include statement
- mode = :lib: returns .lib statement with {corner} section

Paths are always quoted to handle spaces and special characters.
"""
function generate_template_code(code, mode, archive_url, file_path)
    if mode == :inline
        return code
    elseif mode == :include
        path = archive_url !== nothing ? "\"$(archive_url)#$(file_path)\"" : "\"$(file_path)\""
        return ".include $(path)"
    elseif mode == :lib
        path = archive_url !== nothing ? "\"$(archive_url)#$(file_path)\"" : "\"$(file_path)\""
        return ".lib $(path) {corner}"
    else
        error("Invalid mode: $mode. Must be :inline, :include, or :lib")
    end
end

"""
    extract_definitions_from_file(filepath::String; lib_sections=String[], device_blacklist=nothing, encoding=enc"UTF-8", simulator=Ngspice(), strict=false) -> (models, subcircuits)

Parse a SPICE/Spectre file and extract model and subcircuit definitions.
Uses automatic file extension detection (.scs = Spectre, others = SPICE).
Handles .include and .lib statements by recursively parsing referenced files.

The lib_sections parameter whitelists which .lib sections to process. Empty vector processes all sections.
The simulator parameter controls SPICE lexer behavior (Ngspice(), Hspice(), Pspice()).
The strict parameter enforces dialect-specific rules when true.
"""
function extract_definitions_from_file(filepath::String; lib_sections=String[], device_blacklist=nothing, encoding=enc"UTF-8", simulator=Ngspice(), strict=false)
    # Read file with specified encoding
    content = read(filepath, String, encoding)
    spice_dialect = symbol_from_simulator(simulator)
    ast = SpectreNetlistParser.parse(IOBuffer(content); fname=filepath, implicit_title=false, spice_dialect, strict)
    if ast.ps.errored
        #SpectreNetlistParser.visit_errors(ast)
    end

    config = ExtractionConfig(
        includepaths=[dirname(abspath(filepath))],
        lib_sections=lib_sections,
        device_blacklist=device_blacklist,
        current_file=filepath
    )

    # Initialize local result vectors
    models = Vector{Any}()
    subcircuits = Vector{Any}()
    error_stats = Dict{String, Int}()
    failed_files = String[]
    files_processed = 1  # Start with the main file

    try
        extract_definitions(ast, config, models, subcircuits, error_stats, failed_files)

        # Count additional files processed through includes
        files_processed += length(config.parsed_files)

        return (
            models = models,
            subcircuits = subcircuits,
            error_stats = error_stats,
            failed_files = failed_files,
            files_processed = files_processed
        )
    catch e
        showerror(stderr, e)
        println(stderr)
        # throw(e)
        error_type = string(typeof(e))
        error_stats[error_type] = get(error_stats, error_type, 0) + 1
        push!(failed_files, filepath)

        return (
            models = models,
            subcircuits = subcircuits,
            error_stats = error_stats,
            failed_files = failed_files,
            files_processed = files_processed
        )
    end
end

"""
    extract_definitions(ast, config::ExtractionConfig, models::Vector{Any}, subcircuits::Vector{Any}, error_stats::Dict{String,Int}, failed_files::Vector{String})

Extract model and subcircuit definitions from a SPICE/Spectre AST using the provided configuration.
Accumulates results into the provided models and subcircuits vectors.
Accumulates error statistics into error_stats and failed_files vectors.
Returns models as (name, type, subtype, code) tuples and subcircuits as (name, ports, parameters, code) tuples.
For models, subtype indicates polarity (:pmos/:nmos), defaults to :nmos unless 'pchan' or 'pmos' found in parameters.
For subcircuits, parameters is a list of parameter names from both subckt line and .param statements inside.
Code is the full source text extracted using String(node).
"""
function extract_definitions(ast, config::ExtractionConfig, models::Vector{Any}, subcircuits::Vector{Any}, error_stats::Dict{String,Int}, failed_files::Vector{String})
    for stmt in ast.stmts
        if isa(stmt, SNode{SPICENetlistSource})
            # Recurse into netlist source nodes
            extract_definitions(stmt, config, models, subcircuits, error_stats, failed_files)
        elseif isa(stmt, SNode{SpectreNetlistSource})
            # Recurse into spectre netlist source nodes
            extract_definitions(stmt, config, models, subcircuits, error_stats, failed_files)
        elseif isa(stmt, SNode{SP.LibStatement})
            # Recurse into .lib sections, but filter by lib_sections whitelist if specified
            section_name = String(stmt.name)
            if isempty(config.lib_sections) || section_name in config.lib_sections
                extract_definitions(stmt, deeper(config), models, subcircuits, error_stats, failed_files)
            end
        elseif isa(stmt, SNode{SP.IncludeStatement})
            # Handle .include statements
            str = strip(String(stmt.path), ['"', '\''])
            try
                path = resolve_includepath(str, config.includepaths)
                sa = get!(() -> SpectreNetlistParser.parsefile(path; implicit_title=false), config.parsed_files, path)
                new_includepaths = [dirname(path), config.includepaths...]
                extract_definitions(sa, deeper(config; new_includepaths, new_file=path), models, subcircuits, error_stats, failed_files)
            catch e
                error_type = string(typeof(e))
                error_stats[error_type] = get(error_stats, error_type, 0) + 1
                push!(failed_files, str)
                print("Warning: Could not process include $str: ")
                showerror(stdout, e)
                println()
            end
        elseif isa(stmt, SNode{SP.LibInclude})
            # Handle .lib statements (includes with section)
            str = strip(String(stmt.path), ['"', '\''])
            section = String(stmt.name)
            try
                path = resolve_includepath(str, config.includepaths)
                lib_key = (path, section)
                if lib_key ∉ config.libraries
                    push!(config.libraries, lib_key)
                    p = get!(() -> SpectreNetlistParser.parsefile(path; implicit_title=false), config.parsed_files, path)
                    sa = extract_section_from_lib(p; section)
                    if sa !== nothing
                        new_includepaths = [dirname(path), config.includepaths...]
                        extract_definitions(sa, deeper(config; new_includepaths, new_file=path), models, subcircuits, error_stats, failed_files)
                    else
                        println("Warning: Unable to find section '$section' in $str")
                    end
                end
            catch e
                error_type = string(typeof(e))
                error_stats[error_type] = get(error_stats, error_type, 0) + 1
                push!(failed_files, str)
                print("Warning: Could not process lib include $str section $section: ")
                showerror(stdout, e)
                println()
            end
        elseif isa(stmt, SNode{SP.Model})
            name = LSymbol(stmt.name)
            # Check if device name matches blacklist pattern
            if config.device_blacklist !== nothing && occursin(config.device_blacklist, String(name))
                continue
            end
            typ = LSymbol(stmt.typ)
            # Check for PMOS indicators in parameters
            params = [LSymbol(p.name) for p in stmt.parameters]
            subtype = (:pchan in params || :pmos in params) ? :pmos : :nmos
            # Use fullcontents to capture preceding comments
            code = fullcontents(stmt)
            push!(models, (name, typ, subtype, code))
        elseif isa(stmt, SNode{SC.Model})
            name = LSymbol(stmt.name)
            # Check if device name matches blacklist pattern
            if config.device_blacklist !== nothing && occursin(config.device_blacklist, String(name))
                continue
            end
            typ = LSymbol(stmt.master_name)
            # Check for PMOS indicators in parameters
            params = [LSymbol(p.name) for p in stmt.parameters]
            subtype = (:pchan in params || :pmos in params) ? :pmos : :nmos
            # Use fullcontents to capture preceding comments
            code = fullcontents(stmt)
            push!(models, (name, typ, subtype, code))
        elseif isa(stmt, SNode{SP.Subckt})
            name = LSymbol(stmt.name)
            # Check if device name matches blacklist pattern
            if config.device_blacklist !== nothing && occursin(config.device_blacklist, String(name))
                continue
            end
            ports = [LSymbol(node) for node in stmt.subckt_nodes]
            # Start with parameters from subckt line
            params = [LSymbol(p.name) for p in stmt.parameters]
            # Add parameters from .param statements inside subcircuit
            extract_subckt_params!(stmt, params)
            # Use fullcontents to capture preceding comments
            code = fullcontents(stmt)
            push!(subcircuits, (name, ports, params, code))
        elseif isa(stmt, SNode{SC.Subckt})
            name = LSymbol(stmt.name)
            # Check if device name matches blacklist pattern
            if config.device_blacklist !== nothing && occursin(config.device_blacklist, String(name))
                continue
            end
            ports = [LSymbol(node) for node in stmt.subckt_nodes.nodes]
            params = Symbol[]
            # Add parameters from .param statements inside subcircuit
            extract_subckt_params!(stmt, params)
            # Use fullcontents to capture preceding comments
            code = fullcontents(stmt)
            push!(subcircuits, (name, ports, params, code))
        end
    end
    return models, subcircuits
end

function extract_subckt_params!(subckt, params)
    for stmt in subckt.stmts
        if isa(stmt, SNode{SP.ParamStatement})
            for par in stmt.params
                push!(params, LSymbol(par.name))
            end
        elseif isa(stmt, SNode{SC.Parameters})
            for par in stmt.params
                push!(params, LSymbol(par.name))
            end
        end
    end
end

"""
    resolve_includepath(path, includepaths) -> fullpath

Resolve an include path by searching in includepaths.
Returns the full path to the file if found, or errors if not found.

Search order:
1. Check if path exists as-is (absolute or relative to cwd)
2. Search through each directory in includepaths
"""
function resolve_includepath(path, includepaths)
    isfile(path) && return path
    for base in includepaths
        fullpath = joinpath(base, path)
        isfile(fullpath) && return fullpath
    end
    error("include path $path not found in $includepaths")
end

"""
    extract_section_from_lib(p; section) -> SNode or nothing

Extract a specific .lib section from a parsed SPICE file.
"""
function extract_section_from_lib(p; section)
    for node in p.stmts
        if isa(node, SNode{SP.LibStatement})
            if lowercase(String(node.name)) == lowercase(section)
                return node
            end
        elseif isa(node, SNode{SPICENetlistSource})
            # Recurse into netlist source nodes
            return extract_section_from_lib(node; section)
        end
    end
    return nothing
end

"""
    to_mosaic_format(models, subcircuits; source_file=nothing, base_category=String[], mode=:inline, archive_url=nothing, file_device_type=nothing, target_simulators=AbstractSimulator[])

Convert extracted SPICE/Spectre definitions to Mosaic model database format.

Returns a vector of model definitions in CouchDB format with _id keys.
Each model/subcircuit becomes an entry with a generated _id.

Parameters:
- models: Vector of (name, type, subtype, code) tuples from extract_definitions
- subcircuits: Vector of (name, ports, parameters, code) tuples from extract_definitions
- source_file: Optional source filename for metadata
- base_category: Base category path as vector of strings
- mode: Either :inline (embed code directly), :include (use .include), or :lib (use .lib with {corner})
- archive_url: For include/lib modes, the archive URL in zipurl#archive/path format
- file_device_type: Override device type for all subcircuits in this file (e.g., "capacitor")
- target_simulators: Vector of simulators to generate converted templates for (e.g., [Ngspice()]). Only works with :inline mode.

Returns Vector{Dict} with model definitions including _id keys.
"""
function to_mosaic_format(models, subcircuits; source_file=nothing, base_category=String[], mode=:inline, archive_url=nothing, file_device_type=nothing, target_simulators=AbstractSimulator[])
    result = Vector{Dict{String, Any}}()
    
    # Convert models (SPICE .model statements)
    for (name, typ, subtype, code) in models
        try
            # Create unique random ID for this model
            model_id = "models:$(string(uuid4()))"
            
            # Map SPICE device types directly to Mosaic types, using subtype for polarity
            mosaic_type = device_type_mapping(typ, subtype)

            # Generate template code based on mode
            template_code = generate_template_code(code, mode, archive_url, source_file)

            # Build templates array: default first, then dialect-specific conversions
            spice_templates = [Dict(
                "name" => "default",
                "code" => template_code,
                "use-x" => false
            )]

            # Add simulator-specific conversions (only for :inline mode)
            if mode == :inline && !isempty(target_simulators)
                for sim in target_simulators
                    try
                        # Parse the code and convert to target simulator
                        ast = SpectreNetlistParser.parse(IOBuffer(code); start_lang=:spice, implicit_title=false)
                        converted_code = generate_code(ast, sim)

                        push!(spice_templates, Dict(
                            "name" => string(symbol_from_simulator(sim)),
                            "code" => converted_code,
                            "use-x" => false
                        ))
                    catch e
                        println("  Warning: Failed to convert model $name to $(typeof(sim)) simulator: $e")
                    end
                end
            end

            model_def = Dict{String, Any}(
                "_id" => model_id,
                "name" => string(name),
                "type" => mosaic_type,
                "category" => base_category,  # Put models in Models subcategory
                # SPICE models define device templates, not schematic circuits
                "templates" => Dict(
                    "spice" => spice_templates,
                    "spectre" => Vector{Dict{String,Any}}(),
                    "verilog" => Vector{Dict{String,Any}}(),
                    "vhdl" => Vector{Dict{String,Any}}()
                ),
                # TODO: Extract parameter info from model statement
                "props" => Vector{Dict{String,Any}}()
            )
            
            # Add source file info if available
            if source_file !== nothing
                model_def["category"] = vcat(base_category, [basename(source_file)])
            end
            
            push!(result, model_def)
        catch e
            showerror(stderr, e)
            println(stderr)
            continue
        end
    end
    
    # Convert subcircuits  
    for (name, ports, parameters, code) in subcircuits
        try
            # Use file-level override if specified, otherwise detect from name
            device_type = file_device_type !== nothing ? file_device_type : detect_device_type_from_name(name)

            # Create unique random ID for this subcircuit
            model_id = "models:$(string(uuid4()))"
            
            # Determine port layout using heuristics based on port names
            port_layout = determine_port_layout(ports)

            # Generate template code based on mode
            template_code = generate_template_code(code, mode, archive_url, source_file)

            # Build templates array: default first, then dialect-specific conversions
            spice_templates = [Dict(
                "name" => "default",
                "code" => template_code,
                "use-x" => true  # subcircuits always use X prefix
            )]

            # Add simulator-specific conversions (only for :inline mode)
            if mode == :inline && !isempty(target_simulators)
                for sim in target_simulators
                    try
                        # Parse the code and convert to target simulator
                        ast = SpectreNetlistParser.parse(IOBuffer(code); start_lang=:spice, implicit_title=false)
                        converted_code = generate_code(ast, sim)

                        push!(spice_templates, Dict(
                            "name" => string(symbol_from_simulator(sim)),
                            "code" => converted_code,
                            "use-x" => true  # subcircuits always use X prefix
                        ))
                    catch e
                        println("  Warning: Failed to convert subcircuit $name to $(typeof(sim)) simulator: $e")
                    end
                end
            end

            model_def = Dict{String, Any}(
                "_id" => model_id,
                "name" => string(name),
                "type" => device_type,
                "category" => base_category,
                "ports" => port_layout,
                "templates" => Dict(
                    "spice" => spice_templates,
                    "spectre" => Vector{Dict{String,Any}}(),
                    "verilog" => Vector{Dict{String,Any}}(),
                    "vhdl" => Vector{Dict{String,Any}}()
                ),
                # Convert parameter names to props format
                "props" => [Dict("name" => string(param), "tooltip" => "") for param in parameters]
            )
            
            # Add source file info if available  
            if source_file !== nothing
                model_def["category"] = vcat(base_category, [basename(source_file)])
            end
            
            push!(result, model_def)
        catch e
            showerror(stderr, e)
            println(stderr)
            continue
        end
    end
    
    return result
end

# Helper function to map SPICE device types directly to Mosaic types
"""
    detect_device_type_from_name(name::Symbol) -> String

Detect the actual device type from a subcircuit name using common naming patterns.
Returns the Mosaic device type (e.g., "nmos", "pmos", "resistor", etc.) or "ckt" for generic circuits.

Common patterns detected:
- MOSFET: names containing "nfet", "pfet", "nmos", "pmos"  
- BJT: names containing "npn", "pnp", "bjt"
- Resistors: names containing "res", "resistor", "nplus", "pplus", "poly", "rm", "tm"
- Capacitors: names containing "cap", "capacitor", "mim", or various vendor-specific part numbers
- Diodes: names containing "diode", "dio"
"""
function detect_device_type_from_name(name::Symbol)
    name_str = lowercase(string(name))
    
    # MOSFET patterns
    if occursin("nfet", name_str) || occursin("nmos", name_str)
        return "nmos"
    elseif occursin("pfet", name_str) || occursin("pmos", name_str)  
        return "pmos"
    
    # BJT patterns
    elseif occursin("npn", name_str)
        return "npn"
    elseif occursin("pnp", name_str)
        return "pnp"
    elseif occursin("bjt", name_str)
        return "npn"  # Default to NPN for generic BJT
    
    # Resistor patterns - more specific patterns only
    elseif occursin("res", name_str) || occursin("resistor", name_str) || 
           occursin("nplus", name_str) || occursin("pplus", name_str) || 
           occursin("nwell", name_str) || occursin("poly", name_str) ||
           occursin("rm", name_str) || occursin("tm", name_str)
        return "resistor"
    
    # Capacitor patterns - more specific patterns only  
    elseif occursin("cap", name_str) || occursin("capacitor", name_str) ||
           occursin("mim", name_str)
        return "capacitor"
    
    # Diode patterns - more specific patterns only
    elseif occursin("diode", name_str) || occursin("dio", name_str)
        return "diode"
    
    # Inductor patterns - more specific patterns only
    elseif occursin("ind", name_str) || occursin("inductor", name_str)
        return "inductor"
    
    # Default to generic circuit
    else
        return "ckt"
    end
end

"""
    determine_port_layout(ports::Vector{Symbol}) -> Dict{String, Vector{String}}

Determine the layout of ports based on naming patterns and heuristics.

Port placement rules:
- LEFT: ports containing "in", "fb", "ref", "adj", "en", "enable"  
- RIGHT: ports containing "out"
- TOP: "vcc", "vdd", "v+", "hv", "vb", "hb"
- BOTTOM: "gnd", "ground", "com", "vss", "vee", "v-"
- Unassigned ports: split in half between left and right
"""
function determine_port_layout(ports::Vector{Symbol})
    layout = Dict(
        "top" => String[],
        "bottom" => String[], 
        "left" => String[],
        "right" => String[]
    )
    
    unassigned = Symbol[]
    
    for port in ports
        port_str = lowercase(string(port))
        
        # Check for clear functional patterns first
        if contains(port_str, "in") || port_str in ["fb", "ref", "adj", "en", "enable"]
            push!(layout["left"], string(port))
        elseif contains(port_str, "out")
            push!(layout["right"], string(port))
        elseif port_str in ["vcc", "vdd", "v+", "hv", "vb", "hb"]
            push!(layout["top"], string(port))
        elseif port_str in ["gnd", "ground", "com", "vss", "vee", "v-"]
            push!(layout["bottom"], string(port))
        else
            push!(unassigned, port)
        end
    end
    
    # Distribute unassigned ports by splitting in half between left and right
    if !isempty(unassigned)
        mid_point = div(length(unassigned), 2)
        append!(layout["left"], string.(unassigned[1:mid_point]))
        append!(layout["right"], string.(unassigned[mid_point+1:end]))
    end
    
    return layout
end

# Helper function to map SPICE device types directly to Mosaic types
function device_type_mapping(spice_type::Symbol, subtype::Symbol=spice_type)
    spice_str = lowercase(string(spice_type))
    
    # Basic passive components
    if spice_str in ["r", "res"]
        return "resistor"
    elseif spice_str == "c"
        return "capacitor"  
    elseif spice_str == "l"
        return "inductor"
    
    # Diodes
    elseif spice_str == "d"
        return "diode"
    
    # BJT transistors
    elseif spice_str == "npn"
        return "npn"
    elseif spice_str == "pnp"
        return "pnp"
    
    # MOSFET transistors
    elseif spice_str == "nmos"
        return "nmos"
    elseif spice_str == "pmos" 
        return "pmos"
    elseif spice_str == "vdmos"
        return subtype == :pmos ? "pmos" : "nmos"  # Use detected polarity
    
    # JFET transistors - map to BJT as reasonable fallback
    elseif spice_str == "njf"
        return "npn"
    elseif spice_str == "pjf"
        return "pnp"
    
    # MESFET transistors - map to MOSFET as reasonable fallback
    elseif spice_str == "nmf"
        return "nmos"
    elseif spice_str == "pmf"
        return "pmos"
    
    # Unsupported types
    elseif spice_str in ["sw", "csw", "urc", "ltra", "vswitch"]
        error("Unsupported SPICE model type: $spice_str. Switches and transmission lines are not supported in Mosaic format.")
    
    # Unknown types
    else
        error("Unknown SPICE model type: $spice_str. Supported types are: R, RES, C, L, D, NPN, PNP, NMOS, PMOS, VDMOS, NJF, PJF, NMF, PMF")
    end
end

"""
    process_archive(config::ArchiveConfig)

Process an archive using the specified configuration.
"""
function process_archive(config::ArchiveConfig)
    result = Vector{Dict{String, Any}}()

    # Track statistics
    error_stats = Dict{String, Int}()
    failed_files = String[]
    files_processed = 0

    # Create temporary directory for extraction
    temp_dir = mktempdir()
    
    try
        # Download archive
        println("Downloading archive from $(config.url)...")
        archive_file = joinpath(temp_dir, "archive")
        Downloads.download(config.url, archive_file)
        
        # Try to extract archive using p7zip (handles zip, tar.gz, 7z, etc.)
        extract_dir = joinpath(temp_dir, "extracted")
        mkdir(extract_dir)
        
        # Find files to process
        matching_files = Pair{String,String}[]  # full_path => relative_path
        is_archive = false

        # Use p7zip to extract
        p7zip_exe = p7zip_jll.p7zip_path

        try
            println("Extracting archive...")
            run(`$p7zip_exe x $archive_file -o$extract_dir -y`)
            is_archive = true

            # Archive extraction succeeded - find files to process
            if config.entrypoints === nothing
                # Auto-discover SPICE files by walking directory tree
                spice_extensions = [".mod", ".sp", ".lib", ".cir", ".inc", ".txt"]

                for (root, dirs, files) in walkdir(extract_dir)
                    for file in files
                        _, ext = splitext(lowercase(file))
                        if ext in spice_extensions
                            full_path = joinpath(root, file)
                            relative_path = relpath(full_path, extract_dir)
                            push!(matching_files, full_path => relative_path)
                        end
                    end
                end
            else
                # Use specific relative paths provided by user
                for relative_path in config.entrypoints
                    full_path = joinpath(extract_dir, relative_path)
                    if isfile(full_path)
                        push!(matching_files, full_path => relative_path)
                    else
                        println("Warning: specified file not found: $relative_path")
                    end
                end
            end

        catch ProcessFailedException
            # Not an archive - treat downloaded file as single SPICE file
            println("Not an archive, processing as bare SPICE file")
            push!(matching_files, archive_file => basename(config.url))
        end
        
        println("Found $(length(matching_files)) matching files")
        
        # Process each matching file
        for (full_path, relative_path) in matching_files
            println("Processing $relative_path...")
            files_processed += 1

            try
                # Extract definitions from file
                extraction_result = extract_definitions_from_file(full_path; lib_sections=config.lib_sections, device_blacklist=config.device_blacklist, encoding=config.encoding, simulator=config.simulator, strict=config.strict)

                # Merge error statistics
                for (error_type, count) in extraction_result.error_stats
                    error_stats[error_type] = get(error_stats, error_type, 0) + count
                end
                append!(failed_files, extraction_result.failed_files)

                if !isempty(extraction_result.models) || !isempty(extraction_result.subcircuits)
                    # Check for file-level device type override
                    filename = basename(relative_path)
                    file_device_type = get(config.file_device_types, filename, nothing)

                    # Convert to Mosaic format
                    # For archives: use archive_url + relative_path (generates url#path)
                    # For bare files: use nothing + url (generates just url)
                    file_result = to_mosaic_format(
                        extraction_result.models, extraction_result.subcircuits;
                        source_file=is_archive ? relative_path : config.url,
                        base_category=config.base_category,
                        mode=config.mode,
                        archive_url=is_archive ? config.url : nothing,
                        file_device_type=file_device_type,
                        target_simulators=config.target_simulators
                    )

                    # Append results
                    append!(result, file_result)

                    println("  - Found $(length(extraction_result.models)) models, $(length(extraction_result.subcircuits)) subcircuits")
                end
            catch e
                error_type = string(typeof(e))
                error_stats[error_type] = get(error_stats, error_type, 0) + 1
                push!(failed_files, relative_path)
                print("  - Error parsing $relative_path: ")
                showerror(stdout, e)
                println()
            end
        end
        
    finally
        # Clean up temporary directory
        rm(temp_dir, recursive=true, force=true)
    end

    println("Archive processing complete. Generated $(length(result)) total model entries.")

    # Print error statistics if any errors occurred
    if !isempty(error_stats)
        println("\nError Statistics:")
        for (error_type, count) in sort(collect(error_stats), by=x->x[2], rev=true)
            println("  $error_type: $count occurrences")
        end
        println("\nFailed files ($(length(failed_files)) total):")
        for file in failed_files
            println("  - $file")
        end
    end

    return (
        mosaic_models = result,
        error_stats = error_stats,
        failed_files = failed_files,
        files_processed = files_processed,
        models_generated = length(result)
    )
end


export extract_definitions, extract_definitions_from_file, to_mosaic_format, process_archive, ArchiveConfig, ExtractionConfig, detect_device_type_from_name