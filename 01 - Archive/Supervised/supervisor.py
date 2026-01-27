import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from inverse import *
from loss import extract_loss
import torch.optim as optim
from skopt.sampler import Lhs
from sklearn.model_selection import train_test_split

from builder import *


class Supervisor:
    def __init__(self, builder:Builder, model:str):
        self.builder = builder
        self.model = model
        assert model.casefold() in ['doublet', 'triplet'], "Invalid model name - must be 'doublet' or 'triplet'"

        if self.model.casefold() == 'doublet':
            self.nQuads = 2
            self.nDrifts = 3

        if self.model.casefold() == 'triplet':
            self.nQuads = 3
            self.nDrifts = 4

    def _Run(self, i, runDir, nGenerate, cleanup=True):

        # Run BDSIM (temporary file)
        modelPath = f"{self.builder.model_dir}/{self.model.casefold()}_{i}.gmad"
        outfile = f"{runDir}/output-{i}"

        pybdsim.Run.Bdsim(
            gmadpath=modelPath,
            outfile=outfile,
            batch=True,
            seed=1999,
            silent=True,
            ngenerate=nGenerate,
        )

        pybdsim.Run.RebdsimOptics(
            rootpath=outfile + ".root",
            outpath=outfile + "_optics.root",
            silent=True,
        )

        # Extract variables for metrics from RebdsimOptics File
        output_optics = pybdsim.Data.Load(outfile + "_optics.root")
        sigmaX, sigmaY = output_optics.optics.Sigma_x()[-1], output_optics.optics.Sigma_y()[-1]
        sigmaXP, sigmaYP = output_optics.optics.Sigma_xp()[-1], output_optics.optics.Sigma_yp()[-1]
        emittX, emittY = output_optics.optics.Emitt_x()[-1], output_optics.optics.Emitt_y()[-1]
        Nin = output_optics.optics.Npart()[0]
        Nout = output_optics.optics.Npart()[-1]

        # Calculate metrics
        ASigma = abs(sigmaX - sigmaY) / ((sigmaX + sigmaY)/2)
        ASigmaP = abs(sigmaXP - sigmaYP) / ((sigmaXP + sigmaYP)/2)
        AEmitt = abs(emittX - emittY) / ((emittX + emittY)/2)
        T = Nout/Nin
        A = ASigma + ASigmaP + AEmitt

        # Calculate reward from metrics
        R_scale = 100.0
        w_T = 5.0
        w_A = 1.0
        #reward = 10 * (W_T T - W_A tanh(A))
        reward = R_scale * ((w_T * T) - (w_A * np.tanh(A)))


        # Cleanup
        if cleanup:
            os.remove(modelPath)
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{i}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{i}_components.gmad")
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{i}_options.gmad")
            os.remove(f"{self.builder.model_dir}/{self.model.casefold()}_{i}_sequence.gmad")
            os.remove(outfile + ".root")
            os.remove(outfile + "_optics.root")

        return T, A, reward

    def Supervise(self, filename:str, driftLengthMax, nSamples=1000):
        runDir = self.builder.data_dir + "/" + filename
        if not os.path.exists(runDir):
            os.mkdir(runDir)

        nGenerate = self.builder.Options["ngenerate"]

        quadSettings = np.array([0.02, 0.04], dtype=np.float32)
        quadLengths = np.random.choice(quadSettings, size=(nSamples, self.nQuads))

        sampler = Lhs(criterion="maximin", iterations=50)
        space = [(0.0, 1.0)] * (self.nDrifts)
        driftLengths = np.array(sampler.generate(space, nSamples))
        scaledDriftLengths = np.round((driftLengths * driftLengthMax), 3)
        scaledLengths = np.hstack([scaledDriftLengths, quadLengths])

        # Write samples to file
        with open(f"{runDir}/samples.txt", "w") as fin:
            for lengths in scaledLengths:
                fin.write(" ".join(map(str, np.round(lengths, 3))) + "\n")

        nWork = int(os.cpu_count() - 2)

        # Build in Parallel
        if self.model.casefold() == 'doublet':
            with ProcessPoolExecutor(max_workers=nWork) as pool:
                futures = {
                    pool.submit(self.builder.build_pmq_doublet(f"{i}", lengths))
                    for i, lengths in enumerate(scaledLengths, 1)
                }
        if self.model.casefold() == 'triplet':
            with ProcessPoolExecutor(max_workers=nWork) as pool:
                futures = {
                    pool.submit(self.builder.build_pmq_triplet(f"{i}", lengths))
                    for i, lengths in enumerate(scaledLengths, 1)
                }

        # Run in parallel
        labels = []
        with ProcessPoolExecutor(max_workers=nWork) as pool:  # adjust cores
            futures = {
                pool.submit(self._Run, i, runDir, nGenerate, cleanup=True)
                for i, lengths in enumerate(scaledLengths, 1)
            }
            for f in as_completed(futures):
                labels.append(f.result())

        # Save all results
        with open(f"{runDir}/labels.txt", "w") as fout:
            for result in labels:
                fout.write(" ".join(map(str, result)) + "\n")

    def Train(self, filename: str, epochs:int):
        runDir = self.builder.data_dir + "/" + filename
        if not os.path.exists(runDir):
            os.mkdir(runDir)

        X = np.loadtxt(f"{runDir}/samples.txt")
        y = np.loadtxt(f"{runDir}/labels.txt")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1999)

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).float()

        # Apply the Inverse Model
        model = InverseModel(output_dim=y.shape[1], input_dim=X.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        loss_history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred_X = model(y_train)
            loss = loss_fn(pred_X, X_train)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            pred_X_test = model(y_test)
            pred_X_test = torch.clamp(pred_X_test, min=0.0)
            test_loss = loss_fn(pred_X_test, X_test)

        print(f"\n--- Training Complete ---")
        print(f"Final Training Loss: {loss_history[-1]:.6f}")
        print(f"Test Loss: {test_loss.item():.6f}")

        pred_transmissions = []
        true_transmissions = []
        nWork = int(os.cpu_count() - 2)

        with ProcessPoolExecutor(max_workers=nWork) as executor:  # adjust to number of CPU cores
            futures = [
                executor.submit(
                    self.process_sample,
                    i, runDir, X_test, pred_X_test, True
                )
                for i in range(len(X_test))
            ]

            for future in as_completed(futures):
                i, pred_T, true_T, pred_A, true_A = future.result()
                pred_transmissions.append(pred_T)
                true_transmissions.append(true_T)
                print(f"Finished sample {i}: T_pred={pred_T:.4f}, T_true={true_T:.4f}")

        # Then you can plot predicted vs true
        plt.scatter(true_transmissions, pred_transmissions)
        plt.xlabel("True Transmission")
        plt.ylabel("Predicted Transmission")
        plt.plot([0, max(true_transmissions)], [0, max(true_transmissions)], "r--")
        plt.show()

    def process_sample(self, i, runDir, X_test, pred_X_test, cleanup):
        # Convert to numpy and ensure positive
        pred_input = torch.clamp(pred_X_test[i], min=0.0).detach().numpy()
        true_input = X_test[i].detach().numpy()

        self.builder.build_pmq_triplet(f"Pred_{i}", pred_input)
        self.builder.build_pmq_triplet(f"True_{i}", true_input)

        pred_outfile = f"{runDir}/output_Pred_{i}"
        true_outfile = f"{runDir}/output_True_{i}"

        # Run BDSIM + optics
        pybdsim.Run.Bdsim(
            gmadpath=f"{self.builder.model_dir}/triplet_Pred_{i}.gmad",
            outfile=pred_outfile,
            batch=True,
            seed=1999 + i,  # different seed per run
            silent=True,
            ngenerate=self.builder.Options["ngenerate"],
        )
        pybdsim.Run.RebdsimOptics(
            rootpath=pred_outfile + ".root",
            outpath=pred_outfile + "_optics.root",
            silent=True,
        )

        pybdsim.Run.Bdsim(
            gmadpath=f"{self.builder.model_dir}/triplet_True_{i}.gmad",
            outfile=true_outfile,
            batch=True,
            seed=1999 + i,
            silent=True,
            ngenerate=self.builder.Options["ngenerate"],
        )
        pybdsim.Run.RebdsimOptics(
            rootpath=true_outfile + ".root",
            outpath=true_outfile + "_optics.root",
            silent=True,
        )

        # Extract results
        pred_T, pred_A, pred_reward = extract_loss(pred_outfile)
        true_T, true_A, pred_reward = extract_loss(true_outfile)

        # Cleanup
        if cleanup:
            os.remove(f"{self.builder.model_dir}/triplet_Pred_{i}.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_Pred_{i}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_Pred_{i}_components.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_Pred_{i}_options.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_Pred_{i}_sequence.gmad")
            os.remove(pred_outfile + ".root")
            os.remove(pred_outfile + "_optics.root")

            os.remove(f"{self.builder.model_dir}/triplet_True_{i}.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_True_{i}_beam.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_True_{i}_components.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_True_{i}_options.gmad")
            os.remove(f"{self.builder.model_dir}/triplet_True_{i}_sequence.gmad")
            os.remove(true_outfile + ".root")
            os.remove(true_outfile + "_optics.root")

        return (i, pred_T, true_T, pred_A, true_A)

