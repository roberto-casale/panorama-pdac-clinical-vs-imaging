# PANORAMA PDAC — i dati clinici migliorano il rilevamento del tumore su CT?

Studio riproducibile: per il rilevamento del **PDAC** (tumore del pancreas) su TC, confrontiamo
modelli **solo-immagine** vs **immagine + sesso + età**, su 8 estrattori di feature (radiomica
`pyradiomics` + 7 *foundation model*: Merlin, CT-FM, MedicalNet, Models Genesis, STU-Net,
ImageNet, RadImageNet). Dataset: [PANORAMA](https://panorama.grand-challenge.org/).

> **Questa repo contiene solo CODICE + GUIDA.** I dati immagine (licenza **CC BY-NC 4.0**) e le
> feature/pesi rigenerabili **non** sono inclusi: si scaricano/ricreano seguendo le istruzioni.

**Risultato (pilota, 326 casi):** il foundation model di CT addominale **Merlin** è il miglior
estrattore-immagine (AUROC ~0.90); aggiungere sesso+età dà un miglioramento **trascurabile e non
significativo** — coerente su tutti gli estrattori. (La risposta definitiva richiede il dataset completo.)

---

Guida passo-passo per **ricreare l'intero studio da zero su un altro computer**. Tutto avviene
in **4 Jupyter notebook autoconsistenti**: ogni notebook contiene *tutta* la logica al suo
interno (nessun file `.py` da inseguire), spiegata cella per cella.

> **La domanda:** per ogni "estrattore di immagine" (radiomica + 7 foundation model) confrontiamo
> un modello *solo-immagine* con uno *immagine + sesso + età*, per capire se i dati clinici
> aggiungano valore statistico. Coorte pilota: **326 TC olandesi (RUMC + UMCG)**.

---

## Cosa serve
- **Windows** (queste istruzioni). **GPU NVIDIA** consigliata (qui: RTX 5000 Ada, CUDA 12.4).
- **Due versioni di Python**: `winget install Python.Python.3.12` e `winget install Python.Python.3.7`.
  - Perché due? Gli embedding dei foundation model girano in **Python 3.12** (PyTorch), ma
    **pyradiomics** ha wheel solo per **Python 3.7**. Sono due kernel diversi; il "ponte" tra
    loro sono i file `.parquet` in `cache_features/`.
- **git** e connessione internet (per scaricare pesi dei modelli e dataset).

## Struttura delle cartelle
Dopo `git clone`, la cartella radice è `panorama-pdac-clinical-vs-imaging/` (i notebook
rilevano da soli la radice, cercando `imagesTr/` + il file clinico). Le voci *(git-ignored)*
non sono nella repo: si creano/scaricano seguendo la guida.
```
panorama-pdac-clinical-vs-imaging/    ← RADICE del progetto (la cartella clonata)
├─ README.md                          ← questo file
├─ panorama_download.ipynb            ← notebook di DOWNLOAD del dataset
├─ requirements.txt                   ← dipendenze per il notebook di download
├─ imagesTr/  labelsTr/               ← TC + maschere          (dal download · git-ignored)
├─ cache/clinical_information.xlsx    ← metadati clinici        (dal download · git-ignored)
├─ .venv-ml/                          ← ambiente Python 3.12    (da creare · git-ignored)
├─ .venv-radiomics/                   ← ambiente Python 3.7     (da creare · git-ignored)
└─ ml_study/
   ├─ notebooks/                      ← ★ I 4 NOTEBOOK (tutta la logica è qui)
   │  ├─ 1_estrazione_RADIOMICA.ipynb        (kernel Python 3.7)
   │  ├─ 2_estrazione_DEEP.ipynb             (kernel .venv-ml)
   │  ├─ 3_analisi_holdout_DeLong.ipynb      (kernel .venv-ml)
   │  └─ 4_analisi_CV_NadeauBengio.ipynb     (kernel .venv-ml)
   ├─ requirements-ml.txt / requirements-radiomics.txt
   ├─ cache_features/                 ← feature estratte (.parquet)  [rigenerabili · git-ignored]
   └─ model_weights/                  ← pesi STU-Net (scaricati al volo · git-ignored)
```

---

## Passo 1 — Scaricare il dataset PANORAMA
Col notebook incluso `panorama_download.ipynb` (i dati non sono nel repo: licenza CC BY-NC 4.0):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name panorama-pdac --display-name "PANORAMA PDAC (.venv 3.12)"
```
Apri `panorama_download.ipynb`, kernel **PANORAMA PDAC (.venv 3.12)**, al punto 1 imposta
`MODE="percent"`, `TARGET_PERCENT=20`, `SEED=42`, poi **Run All** (su Windows headless: `PYTHONUTF8=1`).
→ Si popolano `imagesTr/`, `labelsTr/`, `cache/clinical_information.xlsx`.

## Passo 2 — Creare i due ambienti + kernel

**2a) Ambiente ML — Python 3.12** (embedding dei foundation model):
```powershell
py -3.12 -m venv .venv-ml
.venv-ml\Scripts\python -m pip install -U pip
.venv-ml\Scripts\python -m pip install -r ml_study\requirements-ml.txt
# ⚠️ torch PER ULTIMO dalla index CUDA (altrimenti monai installa una build CPU):
.venv-ml\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# dipendenze dei foundation model (--no-deps per non toccare torch):
.venv-ml\Scripts\python -m pip install merlin-vlm==0.0.7 --no-deps
.venv-ml\Scripts\python -m pip install transformers einops nltk peft accelerate sentencepiece protobuf lighter-zoo gdown
.venv-ml\Scripts\python -m ipykernel install --user --name panorama-ml --display-name "PANORAMA ML (.venv-ml)"
```
Verifica: `.venv-ml\Scripts\python -c "import torch; print(torch.cuda.is_available())"` → `True`.

**2b) Ambiente radiomica — Python 3.7** (pyradiomics):
```powershell
py -3.7 -m venv .venv-radiomics
.venv-radiomics\Scripts\python -m pip install -U "pip<24"
.venv-radiomics\Scripts\python -m pip install -r ml_study\requirements-radiomics.txt
.venv-radiomics\Scripts\python -m pip install ipykernel==6.16.2
.venv-radiomics\Scripts\python -m ipykernel install --user --name panorama-radiomica --display-name "PANORAMA Radiomica (.venv 3.7)"
```

## Passo 3 — Eseguire i notebook (in ordine)
Apri in VS Code / Jupyter. **Ogni notebook dice in cima quale kernel usare.** `Run All`.

| # | Notebook | Kernel | Cosa fa |
|---|---|---|---|
| 1 | `1_estrazione_RADIOMICA.ipynb` | **Radiomica (.venv 3.7)** | estrae le feature pyradiomics → `feat_radiomics.parquet` |
| 2 | `2_estrazione_DEEP.ipynb` | **ML (.venv-ml)** | estrae gli embedding dei 7 foundation model (GPU) |
| 3 | `3_analisi_holdout_DeLong.ipynb` | **ML (.venv-ml)** | analisi 70/30 + DeLong |
| 4 | `4_analisi_CV_NadeauBengio.ipynb` | **ML (.venv-ml)** | analisi CV ripetuta + Nadeau-Bengio (**consigliato**) |

- I notebook 1 e 2 sono **resumabili** (se interrotti, ripartono dai casi mancanti) e mettono
  tutto in `cache_features/`. Scaricano i pesi dei modelli la prima volta (HuggingFace; STU-Net
  da Google Drive via `gdown`).
- I notebook 3 e 4 **leggono solo i `.parquet`** (dati puri) e fanno tutta l'analisi inline.

## Passo 4 — Riproducibilità
Ogni notebook fissa `SEED=42` (split, Random Forest) e mette cuDNN in modalità deterministica.
Le feature sono in cache. → **A parità di TC ottieni sempre gli stessi identici risultati**
(verificato: ri-estraendo un caso si riottengono valori identici).

---

## ★ File da COPIARE per ricreare lo stesso risultato su un altro PC

**Rieseguire solo l'ANALISI (veloce, senza GPU né ri-estrazione):**
- `ml_study/notebooks/3_*.ipynb` e `4_*.ipynb`
- `ml_study/cache_features/*.parquet`  (le 8 feature già estratte ← garantiscono numeri identici)
- `cache/clinical_information.xlsx`
- `ml_study/requirements-ml.txt`  (per creare `.venv-ml`)
→ con questi + `.venv-ml`, i notebook 3 e 4 danno **gli stessi identici risultati**.

**Rifare TUTTO da zero (incluse le feature):** copia l'intera cartella `ml_study/` + scarica il
dataset (Passo 1) + crea i due ambienti (Passo 2) + esegui i 4 notebook (Passo 3). I pesi dei
modelli si ri-scaricano da soli.

> `cache_features/` e `model_weights/` sono **rigenerabili**; copiarli è una scorciatoia per
> risultati bit-identici senza dipendere dalla GPU.
