import pybdsim
import numpy as np
from scipy import constants

class Builder:
    def __init__(self, beam:pybdsim.Beam, options:pybdsim.Options, model_dir:str, data_dir:str):
        self.Beam = beam
        self.Options = options
        self.model_dir = model_dir
        self.data_dir = data_dir

        self.n_drifts = 0
        self.n_quads = 0
        self.line = None

        self.b_rho = (1e6 / constants.c
                      * np.sqrt((float(self.Beam["energy"].split("*")[0]) ** 2)
                     - (constants.physical_constants[f"proton mass energy equivalent in MeV"][0] ** 2)))

    # Methods
    def __add_beam(self):
        self.line.AddBeam(self.Beam)

    def __add_options(self):
        self.line.AddOptions(self.Options)
        self.line.AddSampler("all")

    def __add_variable_drift(self, drift_length):
        self.n_drifts += 1
        self.line.AddGap(f"LHA_TR_VAC_DRI_0{self.n_drifts}", length=drift_length)

    def __add_variable_pmq(self, polarity:str, quad_length, is_last=False):
        quad_sign = None
        assert(polarity == "F" or polarity == "D")
        if polarity == "F":
            quad_sign = 1
        elif polarity == "D":
            quad_sign = -1

        pmq_k1 = quad_sign * 120 * (1 / self.b_rho)

        self.n_quads += 1
        self.line.AddECol(f"LHA_TR_DIA_COL_0{int(self.n_quads)}",
                          length=1e-6, xsize=0.01, ysize=0.01, material="Cu")
        self.line.AddQuadrupole(f"LHA_TR_MAG_QUAD_0{int(self.n_quads)}",
                                quad_length, k1=quad_sign*pmq_k1, aper1=0.01)

        if not is_last:
            self.n_drifts += 1
            self.line.AddGap(f"LHA_TR_VAC_DRI_0{self.n_drifts}", length=0.02)

    def __add_halbach_pmq(self, polarity:str, quad_length, aper, is_last=False):
        quad_sign = None
        assert(polarity == "F" or polarity == "D")
        if polarity == "F":
            quad_sign = 1
        elif polarity == "D":
            quad_sign = -1

        pmq_k1 = quad_sign * (1.2 / aper) * (1 / self.b_rho)
        pmq_k1 = np.round(pmq_k1, 3)

        self.n_quads += 1
        self.line.AddECol(f"LHA_TR_DIA_COL_0{int(self.n_quads)}",
                          length=1e-6, xsize=aper, ysize=aper, material="Cu")
        self.line.AddQuadrupole(f"LHA_TR_MAG_QUAD_0{int(self.n_quads)}",
                                quad_length, k1=quad_sign*pmq_k1, aper1=aper)

        if not is_last:
            self.n_drifts += 1
            self.line.AddGap(f"LHA_TR_VAC_DRI_0{self.n_drifts}", length=0.02)

    def build_pmq_doublet(self, fname, lengths):
        self.line = pybdsim.Builder.Machine()
        self.__add_beam()
        self.__add_options()

        self.line.AddGap("LHA_TR_VAC_DRI_00", length=0.02)
        self.__add_variable_pmq("F", lengths[2], is_last=False)
        self.__add_variable_drift(lengths[0])
        self.__add_variable_pmq("D", lengths[3], is_last=True)
        self.__add_variable_drift(lengths[1])
        self.n_drifts += 1
        self.line.AddDrift(f"LHA_TR_VAC_DRI_0{self.n_drifts}", length=0.01, aper1=0.0365)
        #self.Line.AddECol("LHA_TR_DIA_NOZZLE", length=0.05, xsize=0.002, ysize=0.002, xsizeOut=0.00287, ysizeOut=0.00287, material="Cu")
        self.line.AddECol(f"LHA_TR_FINAL_APER", length=1e-6, xsize=0.01, ysize=0.01, material="Cu")

        self.line.Write(f"{self.model_dir}/doublet_{fname}.gmad")

    def build_pmq_triplet(self, fname, lengths):
        self.line = pybdsim.Builder.Machine()
        self.__add_beam()
        self.__add_options()

        self.line.AddGap("LHA_TR_VAC_DRI_00", length=0.02)
        self.__add_variable_pmq("F", lengths[3], is_last=False)
        self.__add_variable_drift(lengths[0])
        self.__add_variable_pmq("D", lengths[4], is_last=False)
        self.__add_variable_drift(lengths[1])
        self.__add_variable_pmq("F", lengths[5], is_last=True)
        self.__add_variable_drift(lengths[2])
        self.n_drifts += 1
        self.line.AddGap(f"LHA_TR_VAC_DRI_0{self.n_drifts}", length=0.01)
        # self.Line.AddECol("LHA_TR_DIA_NOZZLE", length=0.05, xsize=0.002, ysize=0.002, xsizeOut=0.00287, ysizeOut=0.00287, material="Cu")
        self.line.AddECol(f"LHA_TR_FINAL_APER", length=1e-6, xsize=0.01, ysize=0.01, material="Cu")

        self.line.Write(f"{self.model_dir}/triplet_{fname}.gmad")

    def build_halbach_triplet(self, fname, params):
        self.line = pybdsim.Builder.Machine()
        self.__add_beam()
        self.__add_options()

        self.line.AddGap("LHA_TR_VAC_DRI_00", length=0.02)
        self.__add_halbach_pmq("F", quad_length=params[3], aper=params[6], is_last=False)
        self.__add_variable_drift(params[0])
        self.__add_halbach_pmq("D", params[4], aper=params[7], is_last=False)
        self.__add_variable_drift(params[1])
        self.__add_halbach_pmq("F", params[5], aper=params[8], is_last=True)
        self.__add_variable_drift(params[2])
        self.n_drifts += 1
        self.line.AddGap(f"LHA_TR_VAC_DRI_0{self.n_drifts}", length=0.01)
        # self.Line.AddECol("LHA_TR_DIA_NOZZLE", length=0.05, xsize=0.002, ysize=0.002, xsizeOut=0.00287, ysizeOut=0.00287, material="Cu")
        self.line.AddECol(f"LHA_TR_FINAL_APER", length=1e-6, xsize=0.01, ysize=0.01, material="Cu")

        self.line.Write(f"{self.model_dir}/triplet_{fname}.gmad")

    def build_halbach_double_triplet(self, fname, params):
        self.line = pybdsim.Builder.Machine()
        self.__add_beam()
        self.__add_options()

        self.line.AddGap("LHA_TR_VAC_DRI_00", length=0.02)
        self.__add_halbach_pmq("F", quad_length=params[3], aper=params[6], is_last=False)
        self.__add_variable_drift(params[0])
        self.__add_halbach_pmq("D", params[4], aper=params[7], is_last=False)
        self.__add_variable_drift(params[1])
        self.__add_halbach_pmq("F", params[5], aper=params[8], is_last=True)
        self.__add_variable_drift(params[2])

        self.n_drifts += 1
        self.line.AddGap(f"LHA_TR_VAC_DRI_0{self.n_drifts}", length=0.02)

        self.__add_halbach_pmq("F", quad_length=params[12], aper=params[15], is_last=False)
        self.__add_variable_drift(params[9])
        self.__add_halbach_pmq("D", params[13], aper=params[16], is_last=False)
        self.__add_variable_drift(params[10])
        self.__add_halbach_pmq("F", params[14], aper=params[17], is_last=True)
        self.__add_variable_drift(params[11])
        self.n_drifts += 1
        self.line.AddGap(f"LHA_TR_VAC_DRI_0{self.n_drifts}", length=0.02)
        # self.Line.AddECol("LHA_TR_DIA_NOZZLE", length=0.05, xsize=0.002, ysize=0.002, xsizeOut=0.00287, ysizeOut=0.00287, material="Cu")
        self.line.AddECol(f"LHA_TR_FINAL_APER", length=1e-6, xsize=0.01, ysize=0.01, material="Cu")

        self.line.Write(f"{self.model_dir}/triplet_{fname}.gmad")