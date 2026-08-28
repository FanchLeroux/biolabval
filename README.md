# Bi-O-Edge Wavefront Sensor — Laboratory Validation

This repository contains the Python code used to reproduce the figures presented in the paper on the **The Bi-O-Edge wavefront sensor: laboratory experimental
demonstration**.

The repository contains the analysis and simulation scripts. The experimental data are hosted separately on Zenodo because of their large size.

## Repository

The source code is available on GitHub:

https://github.com/FanchLeroux/biolabval

The data are available on Zenodo Sandbox:

https://sandbox.zenodo.org/records/593052

---

## 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/FanchLeroux/biolabval.git
cd biolabval
```

---

## 2. Install `uv`

This project uses [`uv`](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

### Linux

Run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the terminal, then check that `uv` is available:

```bash
uv --version
```

### Windows

Open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell, then check:

```powershell
uv --version
```

For additional installation methods, see the official `uv` documentation:

https://docs.astral.sh/uv/getting-started/installation/

---

## 3. Set up the Python environment

From the root of the repository, run:

```bash
uv sync
```

This automatically:

- creates the project virtual environment in `.venv`;
- installs the required Python version;
- installs all required dependencies;

The project uses Python 3.13.5

---

## 4. Download the data

The experimental data required to reproduce the figures are available from the Zenodo Sandbox record:

https://sandbox.zenodo.org/records/593052

The dataset is provided as:

```text
data.zip
```

The archive is approximately 8 GB.

### Linux

From the root of the repository, download the archive:

```bash
wget -O data.zip "https://sandbox.zenodo.org/records/593052/files/data.zip?download=1"
```

Extract it:

```bash
unzip data.zip
```

Then remove the archive:

```bash
rm data.zip
```

### Windows

From the root of the repository, open PowerShell and run:

```powershell
curl.exe -L -o data.zip -w "`nSpeed: %{speed_download} bytes/s`nTotal: %{time_total}s`n" "https://sandbox.zenodo.org/records/593052/files/data.zip?download=1"
```

Extract the archive:

```powershell
Expand-Archive -Path "data.zip" -DestinationPath "."
```

Then remove the archive:

```powershell
Remove-Item data.zip
```

### Data directory structure

After extraction, the repository should have the following structure:

```text
biolabval/
├── data/
│   ├── closed_loop/
│   ├── interaction_matrix/
│   ├── linearity/
│   ├── measure_mask/
│   ├── modal_basis/
│   ├── reconstructor/
│   └── turbulence/
├── outputs/
├── scripts/
├── src/
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

**Important:** the `data/` directory structure must be preserved.

Some HDF5 files use **relative external links** to other HDF5 files. Therefore, the files must remain in the directory structure provided in the archive.

---

## 5. Run the scripts

The figure-generation scripts are located in:

```text
scripts/
```

The available scripts include:

```text
scripts/
├── figure_5_polarization_leakage_impact.py
├── figure_6_bio_profile_illustration.py
├── figure_12_interaction_matrix_visual.py
├── figure_13_interaction_matrix_svd.py
├── figure_14_diagonal_fourier_modes_sensitivity.py
├── figure_15_KL_modes_sensitivity.py
├── figure_16_linearity.py
├── figures_11_17_18_19_20_closed_loop.py
├── figures_Dp1_Dp2_pyramid_vs_bioedge_fourier_modes.py
└── figure_Dp3_pyramid_vs_bioedge_KL_modes.py
```

Scripts should be run using `uv run`.

For example:

```bash
uv run python scripts/figure_5_polarization_leakage_impact.py
```

or:

```bash
uv run python scripts/figure_14_diagonal_fourier_modes_sensitivity.py
```

`uv run` automatically uses the project's virtual environment.

## 6. Computational resources required

The scripts `figure_13_interaction_matrix_svd.py`, `figure_16_linearity.py`, `figures_11_17_18_19_20_closed_loop.py`, `figures_Dp1_Dp2_pyramid_vs_bioedge_fourier_modes.py` and `figure_Dp3_pyramid_vs_bioedge_KL_modes.py` require a large amount of RAM and may require to be run on a workstation or computing server with sufficient RAM rather than on a personal laptop.

---

## 7. Output

Generated figures are saved in:

```text
outputs/
```