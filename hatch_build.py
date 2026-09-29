"""Wheel extras from the source tree, when present: the built web app (app/dist, from `npm run build`) as geopulse/ui
and the example events as geopulse/examples (the API and web app list them). The desktop app bundles neither."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class WebAppHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root)
        if (root / "app" / "dist" / "index.html").is_file():
            build_data["force_include"][str(root / "app" / "dist")] = "geopulse/ui"
        if (root / "examples").is_dir():
            build_data["force_include"][str(root / "examples")] = "geopulse/examples"
