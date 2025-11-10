#!/usr/bin/env python3
import yaml, array, os
import ROOT
import re

def parse_compression(opt):
    if not opt:
        return None
    algo, _, level = opt.partition(":")
    level = int(level) if level else 4
    # Map string to ROOT enum; default to LZ4 if unknown
    alg_map = {
        "LZ4": ROOT.ROOT.kLZ4,
        "ZLIB": ROOT.ROOT.kZLIB,
        "LZMA": ROOT.ROOT.kLZMA,
        "ZSTD": ROOT.ROOT.kZSTD,
        "LZ4HC": ROOT.ROOT.kLZ4HC,
    }
    return alg_map.get(algo.upper(), ROOT.ROOT.kLZ4), level

def open_tree(path_or_tfile, tree_path):
    """
    Supports:
      - tfile + 'dir/Tree'
      - 'file.root:dir/Tree' (inline file path)
    """
    if ":" in tree_path and os.path.exists(tree_path.split(":")[0]):
        fpath, tpath = tree_path.split(":", 1)
        tf = ROOT.TFile.Open(fpath, "READ")
        if not tf or tf.IsZombie():
            raise RuntimeError(f"Cannot open input file: {fpath}")
        obj = tf.Get(tpath)
        if not obj or not isinstance(obj, ROOT.TTree):
            raise RuntimeError(f"Tree '{tpath}' not found in '{fpath}'")
        return tf, obj  # keep file alive
    obj = path_or_tfile.Get(tree_path)
    if not obj or not isinstance(obj, ROOT.TTree):
        raise RuntimeError(f"Tree '{tree_path}' not found in '{path_or_tfile.GetName()}'")
    return None, obj

def make_axis_tuple_1d(name, title, ax):
    if "edges" in ax:
        edges = array.array('d', [float(x) for x in ax["edges"]])
        return (name, title, len(edges)-1, edges)
    return (name, title, int(ax["bins"]), float(ax["min"]), float(ax["max"]))

def make_axis_tuple_2d(name, title, axx, axy):
    # variable or fixed per axis
    if "edges" in axx and "edges" in axy:
        ex = array.array('d', [float(x) for x in axx["edges"]])
        ey = array.array('d', [float(y) for y in axy["edges"]])
        return (name, title, len(ex)-1, ex, len(ey)-1, ey)
    elif "edges" in axx:
        ex = array.array('d', [float(x) for x in axx["edges"]])
        return (name, title, len(ex)-1, ex, int(axy["bins"]), float(axy["min"]), float(axy["max"]))
    elif "edges" in axy:
        ey = array.array('d', [float(y) for y in axy["edges"]])
        return (name, title, int(axx["bins"]), float(axx["min"]), float(axx["max"]), len(ey)-1, ey)
    else:
        return (name, title,
                int(axx["bins"]), float(axx["min"]), float(axx["max"]),
                int(axy["bins"]), float(axy["min"]), float(axy["max"]))

def ensure_dir(tfile, subdir):
    if not subdir:
        return tfile
    # Create nested directories like "a/b/c"
    cur = tfile
    for part in subdir.split("/"):
        d = cur.Get(part)
        if not d:
            cur.mkdir(part)
            d = cur.Get(part)
        cur = d
    return cur

def main(cfg_path):
    ROOT.ROOT.EnableImplicitMT()
    ROOT.TH1.SetDefaultSumw2(True)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    in_file = cfg["input"]
    tf = ROOT.TFile.Open(in_file, "READ")
    if not tf or tf.IsZombie():
        raise RuntimeError(f"Cannot open input file: {in_file}")

    friends_cfg = cfg.get("friends", [])

    for outspec in cfg["outputs"]:
        out_path = outspec["file"]
        out_tf = ROOT.TFile.Open(out_path, "RECREATE")
        if not out_tf or out_tf.IsZombie():
            raise RuntimeError(f"Cannot create output file: {out_path}")

        # optional compression
        comp = parse_compression(outspec.get("compression"))
        if comp:
            algo, level = comp
            out_tf.SetCompressionAlgorithm(algo)
            out_tf.SetCompressionLevel(level)

        alive_files = []  # keep extra TFiles alive for inline file:tree references
        hist_results = [] # list of (subdir, RResultPtr<TH1/2/3/Profile>)

        for sel in outspec["selections"]:
            # Build dataframe from the given tree
            extra_f, tree = open_tree(tf, sel["tree"])
            if extra_f: alive_files.append(extra_f)

            # Attach friends to the TTree before creating the RDataFrame
            for fr in friends_cfg:
                alias = fr.get("alias", "")
                fr_tree = fr["tree"]
                fr_extra_f, fr_tt = open_tree(tf, fr_tree)
                if fr_extra_f: alive_files.append(fr_extra_f)
                if alias:
                    tree.AddFriend(fr_tt, alias)
                else:
                    tree.AddFriend(fr_tt)

            rdf = ROOT.RDataFrame(tree)

            # optional per-selection named defines from YAML
            for d in sel.get("defines", []):
                nm = d.get("name")
                ex = d.get("expr")
                if not nm or not ex:
                    raise ValueError("Each define must have 'name' and 'expr'")
                rdf = rdf.Define(nm, ex)

            
            # --- DEBUG: inspect tree / RDF to find why histos are empty ---
            print(f"[DEBUG] TTree='{tree.GetName()}', TTree.GetEntries()={tree.GetEntries()}")
            try:
                cols = list(rdf.GetColumnNames())
            except Exception:
                cols = [c for c in rdf.GetColumnNames()]
            print("[DEBUG] RDF columns:", cols)
            # materialize Count() to see how many events survive prior Filter/Define
            try:
                n_events = int(rdf.Count().GetValue())
            except Exception as e:
                n_events = f"Count() failed: {e}"
            print(f"[DEBUG] RDF.Count() = {n_events}")
            # print a few sample values for the expressions used in the (first few) histograms
            for h in sel.get("hists", [])[:5]:
                kind = h.get("kind", "H1").upper()
                if kind == "H1":
                    exprs = [h["x"]["expr"]]
                elif kind == "H2":
                    exprs = [h["x"]["expr"], h["y"]["expr"]]
                else:
                    continue
                for ex in exprs:
                    ex = ex.strip()
                    try:
                        # Try AsNumpy() first (fast). If numpy is missing, fall back to TTree.Scan().
                        if re.match(r'^[A-Za-z_]\w*$', ex):
                            try:
                                arr = rdf.AsNumpy([ex]).get(ex)
                                if arr is None:
                                    print(f"[DEBUG] AsNumpy returned no array for '{ex}'")
                                else:
                                    print(f"[DEBUG] AsNumpy({ex}) ->", list(arr[:5]))
                            except Exception as e:
                                print(f"[DEBUG] AsNumpy failed for '{ex}': {e}")
                                try:
                                    print(f"[DEBUG] Fallback: TTree.Scan for '{ex}':")
                                    tree.Scan(ex, "", "", 5)
                                except Exception as e2:
                                    print(f"[DEBUG] TTree.Scan fallback failed: {e2}")
                        else:
                            dbgcol = "__mh_dbg"
                            try:
                                rdf_dbg = rdf.Define(dbgcol, ex)
                                arr = rdf_dbg.AsNumpy([dbgcol]).get(dbgcol)
                                if arr is None:
                                    print(f"[DEBUG] Defined tmp '{dbgcol}' = {ex} but AsNumpy returned no array")
                                else:
                                    print(f"[DEBUG] Defined tmp '{dbgcol}' = {ex}; AsNumpy ->", list(arr[:5]))
                            except Exception as e:
                                print(f"[DEBUG] AsNumpy/Define failed for '{ex}': {e}")
                                try:
                                    print(f"[DEBUG] Fallback: TTree.Scan for '{ex}':")
                                    tree.Scan(ex, "", "", 5)
                                except Exception as e2:
                                    print(f"[DEBUG] TTree.Scan fallback failed: {e2}")
                    except Exception as e:
                        print(f"[DEBUG] Take failed for '{ex}': {e}")
            # --- end debug ---
            
            if sel.get("filter"):
                rdf = rdf.Filter(sel["filter"])
            w_expr = sel.get("weight", "1.0")
            subdir = sel.get("dir", "")
            # RDataFrame histogram weight argument must be a column name.
            # Define a temporary weight column unless weight is the trivial "1.0".
            if w_expr.strip() != "" and w_expr.strip() != "1.0":
                wcol = "__mh_weight"
                rdf = rdf.Define(wcol, w_expr)
            else:
                wcol = None
            # Prepare to define temporary columns for histogram expressions (RDF needs column names)
            expr_to_col = {}
            expr_counter = 0
            def ensure_col_for(expr):
                nonlocal rdf, expr_counter
                expr = expr.strip()
                # if it's already a simple column name, return it directly
                if re.match(r'^[A-Za-z_]\w*$', expr):
                    return expr
                if expr in expr_to_col:
                    return expr_to_col[expr]
                # create a unique column name
                col = f"__mh_expr_{expr_counter}"
                expr_counter += 1
                rdf = rdf.Define(col, expr)
                expr_to_col[expr] = col
                return col


            # Build histograms
            for h in sel["hists"]:
                kind = h.get("kind", "H1").upper()
                name = h["name"]
                title = h.get("title", name)

                if kind == "H1":
                    ax = h["x"]
                    axis_tuple = make_axis_tuple_1d(name, title, ax)
                    col = ensure_col_for(ax["expr"])
                    if wcol:
                        hx = rdf.Histo1D(axis_tuple, col, wcol)
                    else:
                        hx = rdf.Histo1D(axis_tuple, col)
                    hist_results.append((subdir, hx))

                elif kind == "H2":
                    axx, axy = h["x"], h["y"]
                    axis_tuple = make_axis_tuple_2d(name, title, axx, axy)
                    colx = ensure_col_for(axx["expr"])
                    coly = ensure_col_for(axy["expr"])
                    if wcol:
                        hxy = rdf.Histo2D(axis_tuple, colx, coly, wcol)
                    else:
                        hxy = rdf.Histo2D(axis_tuple, colx, coly)
                    hist_results.append((subdir, hxy))


                elif kind == "H3":
                    axx, axy, axz = h["x"], h["y"], h["z"]
                    # Only fixed-binning supported in this minimal example for H3
                    axis_tuple = (name, title,
                                  int(axx["bins"]), float(axx["min"]), float(axx["max"]),
                                  int(axy["bins"]), float(axy["min"]), float(axy["max"]),
                                  int(axz["bins"]), float(axz["min"]), float(axz["max"]))
                    colx = ensure_col_for(axx["expr"])
                    coly = ensure_col_for(axy["expr"])
                    colz = ensure_col_for(axz["expr"])
                    if wcol:
                        hxyz = rdf.Histo3D(axis_tuple, colx, coly, colz, wcol)
                    else:
                        hxyz = rdf.Histo3D(axis_tuple, colx, coly, colz)
                    hist_results.append((subdir, hxyz))

                elif kind in ("P1", "PROFILE1D"):
                    ax = h["x"]
                    y = h["y"]
                    # TProfile via RDataFrame
                    axis_tuple = make_axis_tuple_1d(name, title, ax)
                    colx = ensure_col_for(ax["expr"])
                    coly = ensure_col_for(y["expr"])
                    if wcol:
                        hp = rdf.Profile1D(axis_tuple, colx, coly, wcol)
                    else:
                        hp = rdf.Profile1D(axis_tuple, colx, coly)
                    hist_results.append((subdir, hp))

                else:
                    raise ValueError(f"Unsupported histogram kind: {kind}")

        # Trigger event loop & write histograms into (sub)directories
        for subdir, res in hist_results:
            h = res.GetValue()  # materialize TH1/2/3/TProfile
            out_dir = ensure_dir(out_tf, subdir)
            out_dir.cd()
            h.Write()  # preserves name/title/axes; sumw2 is on

        out_tf.Close()
        # keep input & friend files alive until here
        for f in alive_files:
            f.Close()

    tf.Close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: make_hists.py hists.yaml")
        sys.exit(1)
    main(sys.argv[1])
