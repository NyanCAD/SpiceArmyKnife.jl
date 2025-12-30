import os

def run(settings):
    vafile = os.path.join(settings["source_dir"], "..", "..", "..", "devices", "psp103v4", "psp103.va")
    return settings["run_openvaf"](vafile, "psp103v4.osdi")
    