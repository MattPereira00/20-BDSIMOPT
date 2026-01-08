import pybdsim
from builder import Builder
from bdsim_bayes import BDSIMBayes
from mobo_bdsim import BDSIMMoBO

import time

Beam = pybdsim.Beam.Beam()
Beam.SetParticleType("proton")
Beam.SetEnergy(953.27231, "MeV")
Beam.SetDistributionType("userfile")
Beam._SetDistrFile("../11-Beam/LhARA_0cm_pm2-100k.dat")
Beam._SetDistrFileFormat("x[m]:y[m]:t[ns]:xp[rad]:yp[rad]:E[MeV]")

Options = pybdsim.Options.Options()
Options.SetBeamPipeRadius(10, "mm")
Options.SetBeamPipeThickness(5, "mm")
Options.SetIntegratorSet(integratorSet='"geant4"')
Options.SetSamplerDiameter(0.1, "m")
Options.SetIncludeFringeFields(on=True)
Options.SetGeneralOption("collimatorsAreInfiniteAbsorbers", 1)
Options.SetNGenerate(100000)

if __name__ == "__main__":
    start = time.time()

    Builder = Builder(Beam, Options, "10-Model", "90-BDSIMData")

    #TriOpt = BDSIMBayes(Builder, model="triplet", filename="Triplet_DL1.0_n50_W211_100k_NewLoss_LimitAll")
    #TriOpt.optimise(drift_length_max=1.0, n_calls=50)
    #TriOpt.run100k()

    BDSIMMoBO = BDSIMMoBO(
        Builder,
        model="triplet",
        filename="MOBO_init_full",
        n_initial=24,
        n_iter=40,
        batch_size=4
    )
    BDSIMMoBO.optimise()

    end = time.time()
    print("Time Taken: ", (end - start)/60, "mins")