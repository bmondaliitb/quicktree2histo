import ROOT
f = ROOT.TFile.Open("outTree.root")
t = f.treeAnaWZ
u_pt = ROOT.std.map('string','float')()
print("[Info]:: Printing u_pt keys \n")
t.SetBranchAddress("u_pt", ROOT.AddressOf(u_pt))
t.GetEntry(0)
for i, (key,value) in enumerate(u_pt):
    # print key index also
    print(i, key, value)


print("\n")
print("[Info]:: Printing tu_pt keys \n")
tu_pt = ROOT.std.map('string','float')()
t.SetBranchAddress("tu_pt", ROOT.AddressOf(tu_pt))
t.GetEntry(0)
for i, (key,value) in enumerate(tu_pt):
    print(i, key, value)



