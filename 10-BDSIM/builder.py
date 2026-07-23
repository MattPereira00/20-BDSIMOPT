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

    def __add_segmented_drift(self, base_name, length, n_segments):
        """Splits a drift into n_segments equal parts separated by markers,
        so rebdsimOptics reports optics at several points inside it instead
        of only at its start/end."""
        seg_length = length / n_segments
        for i in range(n_segments):
            self.line.AddDrift(f"{base_name}_seg{i + 1}", length=seg_length)
            if i < n_segments - 1:
                self.line.AddMarker(f"{base_name}_mk{i + 1}")

    def build_s1_gl_diagnostic(self, fname, b1=1.400, b2=0.579, b3=0.817,
                                b4=1.309, b5=0.535, b6=0.785, b7=0.065,
                                n_seg_23=6, n_seg_34=12):
        """
        Same S1GL line as build_s1_gl, but with the GL2->GL3 and GL3->GL4
        drifts split into short segments (with markers between them) so
        rebdsimOptics reports alpha_x/sigma_x at several points inside
        those drifts instead of only at their endpoints.

        Used to check the GL2-GL3 "parallel beam" condition (alpha_x ~ 0
        throughout the drift) and the post-GL3 focus condition (a real
        sigma_x minimum shortly into the GL3->GL4 drift).

        Returns the S positions bounding the GL2->GL3 and GL3->GL4 regions
        so extract_stage1_diagnostics can select exactly those points.
        """
        self.line = pybdsim.Builder.Machine()
        self.__add_beam()
        self.__add_options()

        self.line.AddDrift("StartToGL1", length=0.25)
        self.line.AddGaborLens("GL1", length=0.857, b=b1, aper1=0.1)
        self.line.AddDrift("GL1ToGL2", length=0.3)
        self.line.AddGaborLens("GL2", length=0.857, b=b2, aper1=0.1)

        s_gl2_end = 0.25 + 0.857 + 0.3 + 0.857

        gl2_to_gl3_length = 1.9455
        self.__add_segmented_drift("GL2ToGL3", gl2_to_gl3_length, n_seg_23)

        s_gl3_start = s_gl2_end + gl2_to_gl3_length

        self.line.AddGaborLens("GL3", length=0.857, b=b3, aper1=0.1)

        s_gl3_end = s_gl3_start + 0.857

        gl3_to_gl4_length = 2.8456
        self.__add_segmented_drift("GL3ToGL4", gl3_to_gl4_length, n_seg_34)

        s_gl4_start = s_gl3_end + gl3_to_gl4_length

        self.line.AddGaborLens("GL4", length=0.857, b=b4, aper1=0.1)
        self.line.AddDrift("GL4ToGL5", length=0.8)
        self.line.AddGaborLens("GL5", length=0.857, b=b5, aper1=0.1)
        self.line.AddDrift("GL5ToGL6", length=3.2)
        self.line.AddGaborLens("GL6", length=0.857, b=b6, aper1=0.1)
        self.line.AddDrift("GL6ToGL7", length=0.8)
        self.line.AddGaborLens("GL7", length=0.857, b=b7, aper1=0.1)
        self.line.AddDrift("GL7ToARC", length=2.27)

        self.line.Write(f"{self.model_dir}/S1GL_diag_{fname}.gmad")

        return {
            "gl2_to_gl3": (s_gl2_end, s_gl3_start),
            "gl3_to_gl4": (s_gl3_end, s_gl4_start),
        }

    def build_s1_col1(self, fname, params):
        self.line = pybdsim.Builder.Machine()
        self.__add_beam()
        self.__add_options()
        self.line.AddDrift("StartToGL1", length=0.25)
        self.line.AddGaborLens("GL1", length=0.857, b=1.400, aper1=0.1)
        self.line.AddDrift("GL1ToGL2", length=0.3)
        self.line.AddGaborLens("GL2", length=0.857, b=0.579, aper1=0.1)
        self.line.AddDrift("GL2ToGL3", length=1.9455)
        self.line.AddGaborLens("GL3", length=0.857, b=0.817, aper1=0.1)
        self.line.AddDrift("GL3ToCOL1", length=1.921)
        self.line.AddECol("COL1", length=0.01,
                          xsize=params[0], ysize=params[0],
                          xsizeOut=params[1], ysizeOut=params[1],
                          material="Fe")

        self.line.Write(f"{self.model_dir}/S1_COL1_{fname}.gmad")