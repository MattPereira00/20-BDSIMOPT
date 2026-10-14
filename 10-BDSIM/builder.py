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

        self.line.Write(f"{self.model_dir}/double_triplet_{fname}.gmad")

    def build_s1_gl(self, fname, params):
        self.line = pybdsim.Builder.Machine()
        self.__add_beam()
        self.__add_options()
        self.line.AddDrift("StartToGL1", length=0.25)
        self.line.AddGaborLens("GL1", length=0.857, b=1.400, aper1=0.1)
        self.line.AddDrift("GL1ToGL2", length=0.3)
        self.line.AddGaborLens("GL2", length=0.857, b=0.579, aper1=0.1)
        self.line.AddDrift("GL2ToGL3", length=1.9455)
        self.line.AddGaborLens("GL3", length=0.857, b=0.817, aper1=0.1)
        self.line.AddDrift("GL3ToGL4", length=2.8456)
        self.line.AddGaborLens("GL4", length=0.857, b=params[0], aper1=0.1)
        self.line.AddDrift("GL4ToGL5", length=0.8)
        self.line.AddGaborLens("GL5", length=0.857, b=params[1], aper1=0.1)
        self.line.AddDrift("GL5ToGL6", length=3.2)
        self.line.AddGaborLens("GL6", length=0.857, b=params[2], aper1=0.1)
        self.line.AddDrift("GL6ToGL7", length=0.8)
        self.line.AddGaborLens("GL7", length=0.857, b=params[3], aper1=0.1)
        self.line.AddDrift("GL7ToARC", length=2.27)

        self.line.Write(f"{self.model_dir}/S1GL_{fname}.gmad")

    def __add_capture_and_eneselect(self, rf_voltage, GL3B, col_aper):
        # CAPTURE (GL1, GL2 fixed - matches the S1GL upstream optimization)
        self.line.AddDrift("StartToGL1", length=0.25)
        self.line.AddGaborLens("GL1", length=0.857, b=1.400, aper1=0.1)
        self.line.AddDrift("GL1ToGL2", length=0.3)
        self.line.AddGaborLens("GL2", length=0.857, b=0.579, aper1=0.1)
        self.line.AddDrift("GL2ToCAV1", length=0.15 + 1.0185)  # TR_VAC_DRI_04/05

        # ENESELECT
        self.line.AddRFCavity("CAV1", length=0.26, gradient=rf_voltage / 0.26,
                               frequency=50e6, phase=-np.pi / 2,
                               horizontalWidth=0.3)
        self.line.AddDrift("RFToGL3", length=0.127 + 0.24 + 0.15)  # LEL_VAC_DRI_00/01
        self.line.AddGaborLens("GL3", length=0.857, b=GL3B, aper1=0.1)
        self.line.AddDrift("GL3ToCOL1", length=0.15 + 1.771)  # LEL_VAC_DRI_02/03
        self.line.AddECol("COL1", length=0.01,
                          xsize=col_aper, ysize=col_aper, material="Fe")
        self.line.AddDrift("COL1ToCOL2", length=0.06461 + 0.13539)  # LEL_VAC_DRI_04/05
        self.line.AddDrift("COL2", length=0.01)  # LHA_LEL_DIA_COL_02 (stand-in drift)

    def build_stage1_energy_selection(self, fname, params):
        # params = [rf_voltage, GL3B, col_aper] - matches CAPTURE+ENESELECT
        # from the CCAP-v4.6 lattice element-by-element.
        rf_voltage, GL3B, col_aper = params[0], params[1], params[2]

        self.line = pybdsim.Builder.Machine()
        self.__add_beam()
        self.__add_options()

        self.__add_capture_and_eneselect(rf_voltage, GL3B, col_aper)
        self.line.AddDrift("CAV2", length=0.5)  # LHA_LEL_HRF_CAV_02 (stand-in drift)

        self.line.Write(f"{self.model_dir}/S1_ENESELECT_{fname}.gmad")

    S1GL_B4_B7 = (0.7223, 0.9105, 0.4396, 0.8634)  # CCAP-v4.6 GL4B-GL7B

    def build_stage1_invitro(self, fname, params):
        # params = [rf_voltage, GL3B, col_aper, rf_voltage2, mom_aper] - CAPTURE
        # through the S1 in-vitro line end station, element-by-element from CCAP-v4.6.
        rf_voltage, GL3B, col_aper, rf_voltage2, mom_aper = params[:5]
        b4, b5, b6, b7 = self.S1GL_B4_B7

        self.line = pybdsim.Builder.Machine()
        self.__add_beam()
        self.__add_options()

        self.__add_capture_and_eneselect(rf_voltage, GL3B, col_aper)
        self.line.AddRFCavity("CAV2", length=0.26, gradient=rf_voltage2 / 0.26,
                               frequency=50e6, phase=-np.pi / 2,
                               horizontalWidth=0.3)

        # MATCHING
        self.line.AddDrift("RFToGL4", length=0.0546 + 0.24 + 0.15)  # LEL_VAC_DRI_06/07
        self.line.AddGaborLens("GL4", length=0.857, b=b4, aper1=0.1)
        self.line.AddDrift("GL4ToGL5", length=0.15 + 0.5 + 0.15)  # LEL_VAC_DRI_08/09/10
        self.line.AddGaborLens("GL5", length=0.857, b=b5, aper1=0.1)
        self.line.AddDrift("GL5ToGL6", length=0.15 + 2.9 + 0.15)  # LEL_VAC_DRI_11/12/13
        self.line.AddGaborLens("GL6", length=0.857, b=b6, aper1=0.1)
        self.line.AddDrift("GL6ToGL7", length=0.15 + 0.5 + 0.15)  # LEL_VAC_DRI_14/15/16
        self.line.AddGaborLens("GL7", length=0.857, b=b7, aper1=0.1)
        self.line.AddDrift("GL7ToOCT", length=0.15 + 0.45)  # LEL_VAC_DRI_17/17a
        self.line.AddOctupole("OCT1", length=0.1, k3=0)
        self.line.AddDrift("OCTToSD", length=0.15 + 0.84 + 0.15)  # LEL_VAC_DRI_18, DIP_01_dr, DRI_19

        # S1SDCOL (transverse cleaning collimator is a drift in the lattice)
        self.line.AddDrift("COL3", length=0.05)
        self.line.AddDrift("COL3ToARC", length=0.18 + 0.2)  # LEL_VAC_DRI_19a/20

        # S1INVITRO vertical arc
        tilt = -np.pi / 2
        self.line.AddDipole("ARCDIP1", category="sbend", length=0.8, angle=np.pi / 4, tilt=tilt)
        self.line.AddDrift("DIP1ToQ1", length=0.2)
        self.line.AddQuadrupole("ARCQ1", length=0.1, k1=-22.544, tilt=tilt, aper1=0.05)
        self.line.AddDrift("Q1ToQ2", length=0.4)
        self.line.AddQuadrupole("ARCQ2", length=0.1, k1=31.3768, tilt=tilt, aper1=0.05)
        self.line.AddDrift("Q2ToQ3", length=0.2)
        self.line.AddQuadrupole("ARCQ3", length=0.1, k1=-31.5123, tilt=tilt, aper1=0.05)
        self.line.AddDrift("Q3ToMOMCOL", length=0.2 - 0.005)
        self.line.AddECol("MOMCOL", length=0.01, xsize=0.018, ysize=mom_aper, material="Fe")
        self.line.AddDrift("MOMCOLToQ4", length=0.2 - 0.005)
        self.line.AddQuadrupole("ARCQ4", length=0.1, k1=-31.5123, tilt=tilt, aper1=0.05)
        self.line.AddDrift("Q4ToQ5", length=0.2)
        self.line.AddQuadrupole("ARCQ5", length=0.1, k1=31.3768, tilt=tilt, aper1=0.05)
        self.line.AddDrift("Q5ToQ6", length=0.4)
        self.line.AddQuadrupole("ARCQ6", length=0.1, k1=-22.544, tilt=tilt, aper1=0.05)
        self.line.AddDrift("Q6ToDIP2", length=0.2)
        self.line.AddDipole("ARCDIP2", category="sbend", length=0.8, angle=np.pi / 4, tilt=tilt)
        self.line.AddDrift("DIP2ToEND", length=0.2)  # LEL_VAC_DRI_29
        self.line.AddDrift("ENDSTATION", length=2.0)  # LEL_VAC_DRI_30, drift to end station

        self.line.Write(f"{self.model_dir}/S1_INVITRO_{fname}.gmad")
