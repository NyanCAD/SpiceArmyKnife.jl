# Phase 0: Guard deprecated DAECompiler methods
@static if CedarSim.USE_DAECOMPILER
    import DAECompiler: IRODESystem
    Base.@deprecate IRODESystem(ac::ACSol) get_sys(ac)
    Base.@deprecate IRODESystem(ac::NoiseSol)  get_sys(ac)
    Base.@deprecate CircuitIRODESystem(sol::SciMLBase.AbstractODESolution)  get_sys(sol)
end