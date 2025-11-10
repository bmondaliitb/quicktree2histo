import ROOT
f = ROOT.TFile.Open("outTree.root")
t = f.treeAnaWZ
u_pt = ROOT.std.map('string','float')()
print("[Info]:: Printing u_pt keys \n")
t.SetBranchAddress("u_pt", ROOT.AddressOf(u_pt))
t.GetEntry(0)
for key,value in u_pt:
    print(key, value)


print("\n")
print("[Info]:: Printing tu_pt keys \n")
tu_pt = ROOT.std.map('string','float')()
t.SetBranchAddress("tu_pt", ROOT.AddressOf(tu_pt))
t.GetEntry(0)
for key,value in tu_pt:
    print(key, value)



