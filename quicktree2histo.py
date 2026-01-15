#!/usr/bin/env python3
import argparse
import array
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import ROOT
import yaml


@dataclass
class DefineSpec:
    name: str
    expr: str


@dataclass
class HistogramSpec:
    name: str
    title: str
    kind: str
    axes: Dict[str, Dict[str, Any]]


@dataclass
class SelectionSpec:
    tree: str
    filter_expr: Optional[str]
    weight_expr: str
    directory: str
    defines: List[DefineSpec] = field(default_factory=list)
    histograms: List[HistogramSpec] = field(default_factory=list)


@dataclass
class OutputSpec:
    file: str
    selections: List[SelectionSpec] = field(default_factory=list)


@dataclass
class Config:
    input_file: str
    outputs: List[OutputSpec] = field(default_factory=list)


class ConfigLoader:
    @staticmethod # This function belongs with the class, not to an instance
    def load(path: str) -> Config:
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

        outputs: List[OutputSpec] = []
        for out_cfg in data.get("outputs", []):
            selections: List[SelectionSpec] = []
            for sel_cfg in out_cfg.get("selections", []):
                defines = []
                for d in sel_cfg.get("defines", []):
                    obj = DefineSpec(name=d["name"], expr=d["expr"])
                    defines.append(obj)

                histograms = []
                for h in sel_cfg.get("hists", []):
                    # Extract basic fields
                    name = h["name"]
                    title = h.get("title", h["name"])  # Use name as fallback if title missing
                    kind = h.get("kind", "H1").upper()  # Default to "H1" if kind missing
                    
                    # Filter axes to only keep x, y, z dimensions
                    axes = {}
                    for key, value in h.items():
                        if key in {"x", "y", "z"}:
                            axes[key] = dict(value)
                    
                    # Create the histogram specification object
                    histogram = HistogramSpec(
                        name=name,
                        title=title,
                        kind=kind,
                        axes=axes
                    )
                    histograms.append(histogram)
                
                # Extract selection configuration fields
                tree_name = sel_cfg["tree"]
                filter_expression = sel_cfg.get("filter")  # Optional: can be None
                weight_expression = sel_cfg.get("weight", "1.0")  # Default weight is 1.0
                output_directory = sel_cfg.get("dir", "")  # Default to root directory
                
                # Create the selection specification
                selection = SelectionSpec(
                    tree=tree_name,
                    filter_expr=filter_expression,
                    weight_expr=weight_expression,
                    directory=output_directory,
                    defines=defines,
                    histograms=histograms,
                )
                
                # Add to selections list
                selections.append(selection)

            outputs.append(OutputSpec(file=out_cfg["file"], selections=selections))

        return Config(input_file=data["input"], outputs=outputs)

def print_config(config: Config) -> None:
    """Print the loaded configuration in a structured, readable format."""
    print("\n" + "=" * 60)
    print("LOADED CONFIGURATION")
    print("=" * 60)
    
    print(f"\nInput File: {config.input_file}")
    print(f"Number of Outputs: {len(config.outputs)}\n")
    
    for out_idx, output in enumerate(config.outputs, 1):
        print(f"  Output {out_idx}: {output.file}")
        print(f"    Selections: {len(output.selections)}")
        
        for sel_idx, selection in enumerate(output.selections, 1):
            print(f"      Selection {sel_idx}:")
            print(f"        Tree: {selection.tree}")
            print(f"        Filter: {selection.filter_expr}")
            print(f"        Weight: {selection.weight_expr}")
            print(f"        Directory: {selection.directory}")
            print(f"        Defines: {len(selection.defines)}")
            for define in selection.defines:
                print(f"          - {define.name} = {define.expr}")
            print(f"        Histograms: {len(selection.histograms)}")
            for hist in selection.histograms:
                print(f"          - {hist.name} ({hist.kind}): {hist.title}")
                for axis_name, axis_cfg in hist.axes.items():
                    print(f"              {axis_name}: {axis_cfg}")
    
    print("=" * 60 + "\n")


class AxisFactory:
    @staticmethod
    def make_1d(name: str, title: str, cfg: Dict[str, Any]) -> Tuple[Any, ...]:
        if "edges" in cfg:
            edges = array.array("d", [float(x) for x in cfg["edges"]])
            return name, title, len(edges) - 1, edges
        return name, title, int(cfg["bins"]), float(cfg["min"]), float(cfg["max"])

    @staticmethod
    def make_2d(
        name: str,
        title: str,
        cfg_x: Dict[str, Any],
        cfg_y: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        if "edges" in cfg_x and "edges" in cfg_y:
            ex = array.array("d", [float(x) for x in cfg_x["edges"]])
            ey = array.array("d", [float(y) for y in cfg_y["edges"]])
            return name, title, len(ex) - 1, ex, len(ey) - 1, ey
        if "edges" in cfg_x:
            ex = array.array("d", [float(x) for x in cfg_x["edges"]])
            return name, title, len(ex) - 1, ex, int(cfg_y["bins"]), float(cfg_y["min"]), float(cfg_y["max"])
        if "edges" in cfg_y:
            ey = array.array("d", [float(y) for y in cfg_y["edges"]])
            return name, title, int(cfg_x["bins"]), float(cfg_x["min"]), float(cfg_x["max"]), len(ey) - 1, ey
        return (
            name,
            title,
            int(cfg_x["bins"]),
            float(cfg_x["min"]),
            float(cfg_x["max"]),
            int(cfg_y["bins"]),
            float(cfg_y["min"]),
            float(cfg_y["max"]),
        )

    @staticmethod
    def make_3d(
        name: str,
        title: str,
        cfg_x: Dict[str, Any],
        cfg_y: Dict[str, Any],
        cfg_z: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        return (
            name,
            title,
            int(cfg_x["bins"]),
            float(cfg_x["min"]),
            float(cfg_x["max"]),
            int(cfg_y["bins"]),
            float(cfg_y["min"]),
            float(cfg_y["max"]),
            int(cfg_z["bins"]),
            float(cfg_z["min"]),
            float(cfg_z["max"]),
        )


def ensure_dir(tfile: ROOT.TFile, subdir: str) -> ROOT.TDirectory:
    if not subdir:
        return tfile
    current = tfile
    for part in subdir.split("/"):
        nxt = current.Get(part)
        if not nxt:
            current.mkdir(part)
            nxt = current.Get(part)
        current = nxt
    return current


def open_root_file(path: str) -> ROOT.TFile:
    tf = ROOT.TFile.Open(path, "READ")
    if not tf or tf.IsZombie():
        raise RuntimeError(f"Cannot open input file: {path}")
    return tf


def resolve_tree(tfile: ROOT.TFile, tree_name: str) -> ROOT.TTree:
    obj = tfile.Get(tree_name)
    if not obj or not isinstance(obj, ROOT.TTree):
        raise RuntimeError(f"Tree '{tree_name}' not found in '{tfile.GetName()}'")
    return obj


def build_histograms(
    rdf: ROOT.RDataFrame,
    selection: SelectionSpec,
) -> List[Tuple[str, ROOT.RDF.RResultPtr]]:
    results: List[Tuple[str, ROOT.RDF.RResultPtr]] = []
    expr_cache: Dict[str, str] = {}
    counter = 0

    def ensure_column(expr: str) -> str:
        nonlocal rdf, counter
        expr = expr.strip()
        if expr.isidentifier():
            return expr
        if expr in expr_cache:
            return expr_cache[expr]
        col = f"__mh_expr_{counter}"
        counter += 1
        rdf = rdf.Define(col, expr)
        expr_cache[expr] = col
        return col

    for hist in selection.histograms:
        kind = hist.kind
        axes = hist.axes

        if kind == "H1":
            cfg_x = axes["x"]
            axis = AxisFactory.make_1d(hist.name, hist.title, cfg_x)
            column = ensure_column(cfg_x["expr"])
            results.append(
                (selection.directory, rdf.Histo1D(axis, column, selection.weight_expr) if selection.weight_expr.strip() not in {"", "1.0"}
                 else rdf.Histo1D(axis, column))
            )

        elif kind == "H2":
            cfg_x, cfg_y = axes["x"], axes["y"]
            axis = AxisFactory.make_2d(hist.name, hist.title, cfg_x, cfg_y)
            colx, coly = ensure_column(cfg_x["expr"]), ensure_column(cfg_y["expr"])
            results.append(
                (
                    selection.directory,
                    rdf.Histo2D(axis, colx, coly, selection.weight_expr) if selection.weight_expr.strip() not in {"", "1.0"}
                    else rdf.Histo2D(axis, colx, coly),
                )
            )

        elif kind == "H3":
            cfg_x, cfg_y, cfg_z = axes["x"], axes["y"], axes["z"]
            axis = AxisFactory.make_3d(hist.name, hist.title, cfg_x, cfg_y, cfg_z)
            colx = ensure_column(cfg_x["expr"])
            coly = ensure_column(cfg_y["expr"])
            colz = ensure_column(cfg_z["expr"])
            results.append(
                (
                    selection.directory,
                    rdf.Histo3D(axis, colx, coly, colz, selection.weight_expr) if selection.weight_expr.strip() not in {"", "1.0"}
                    else rdf.Histo3D(axis, colx, coly, colz),
                )
            )

        elif kind in {"P1", "PROFILE1D"}:
            cfg_x, cfg_y = axes["x"], axes["y"]
            axis = AxisFactory.make_1d(hist.name, hist.title, cfg_x)
            colx, coly = ensure_column(cfg_x["expr"]), ensure_column(cfg_y["expr"])
            results.append(
                (
                    selection.directory,
                    rdf.Profile1D(axis, colx, coly, selection.weight_expr) if selection.weight_expr.strip() not in {"", "1.0"}
                    else rdf.Profile1D(axis, colx, coly),
                )
            )

        else:
            raise ValueError(f"Unsupported histogram kind '{hist.kind}'")

    return results


def process_selection(tfile: ROOT.TFile, selection: SelectionSpec) -> List[Tuple[str, ROOT.RDF.RResultPtr]]:
    tree = resolve_tree(tfile, selection.tree)
    rdf = ROOT.RDataFrame(tree)

    for define in selection.defines:
        rdf = rdf.Define(define.name, define.expr)

    if selection.filter_expr:
        rdf = rdf.Filter(selection.filter_expr)

    results = build_histograms(rdf, selection)
    return results


def process_output(input_file: ROOT.TFile, spec: OutputSpec) -> None:
    out_file = ROOT.TFile(spec.file, "RECREATE")
    if not out_file or out_file.IsZombie():
        raise RuntimeError(f"Cannot create output file: {spec.file}")

    try:
        # List to store histogram results before writing to file
        # Each tuple contains: (output_directory_path, histogram_result_object)
        pending: List[Tuple[str, ROOT.RDF.RResultPtr]] = []
        for selection in spec.selections:
            pending.extend(process_selection(input_file, selection))

        for subdir, res in pending:
            histogram = res.GetValue()
            ensure_dir(out_file, subdir).cd()
            histogram.Write()
    finally:
        out_file.Close()


def run(config_path: str) -> None:
    ROOT.ROOT.EnableImplicitMT()
    ROOT.TH1.SetDefaultSumw2(True)

    config = ConfigLoader.load(config_path)
    print_config(config)
    input_file = open_root_file(config.input_file)

    try:
        for output in config.outputs:
            process_output(input_file, output)
    finally:
        input_file.Close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple tree-to-histogram converter")
    parser.add_argument("config", help="YAML configuration file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()