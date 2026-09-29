#!/usr/bin/env bash
# Download the corpora and build every HDF5 dataset for training.
#
# Idempotent: each step leaves a stamp in $DATA/.done/ and is skipped next time,
# so an interrupted run picks up where it stopped. ~33 GB of downloads and
# ~24 GB of HDF5, all on the WSL filesystem -- never /mnt, HDF5 locking fails
# on drvfs.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../wsl_env.sh"
DATA="${DFN_DATA:-/root/dfn-data}"
PREP=/opt/miniforge3/lib/python3.11/site-packages/df/scripts/prepare_data.py
mkdir -p "$DATA"/{vctk,demand,fsd50k,lists,hdf5,.done}
cd "$DATA"

step() {  # $1 name, rest: command. Skipped once it has succeeded.
  local name="$1"; shift
  if [ -e ".done/$name" ]; then echo "--- $name (done)"; return; fi
  echo "### $name  $(date '+%F %T')"
  "$@"
  touch ".done/$name"
}

fetch() {  # $1 file, $2 url. Resumes a .part; a finished file is never re-requested.
  [ -e "$1" ] && return
  curl -fL -C - --retry 5 --retry-delay 5 -s -o "$1.part" "$2"
  mv "$1.part" "$1"
}

dl_vctk() {  # VCTK 0.92, CC BY 4.0, 48 kHz
  fetch vctk/DS_10283_3443.zip "https://datashare.ed.ac.uk/download/DS_10283_3443.zip"
}

dl_demand() {  # DEMAND 48 kHz, CC BY-SA 3.0
  for env in DKITCHEN DLIVING DWASHING NFIELD NPARK NRIVER OHALLWAY OMEETING OOFFICE \
             PCAFETER PRESTO PSTATION SPSQUARE STRAFFIC TBUS TCAR TMETRO; do
    fetch "demand/${env}_48k.zip" "https://zenodo.org/records/1227121/files/${env}_48k.zip" &
  done
  wait
  # `wait` swallows the background exit codes; a leftover .part is the failure.
  ! ls demand/*.part 2>/dev/null
}

dl_fsd() {  # FSD50K dev, per-clip CC-BY / CC0, 44.1 kHz (prepare_data resamples)
  for p in .z01 .z02 .z03 .z04 .z05 .zip; do
    fetch "fsd50k/FSD50K.dev_audio$p" "https://zenodo.org/records/4060432/files/FSD50K.dev_audio$p"
  done
}

unpack_vctk() {
  # The DataShare download is a wrapper zip around the real archive, which has
  # the same name -- so the first unzip looks like a no-op. Two are needed.
  cd vctk
  unzip -q -o DS_10283_3443.zip
  unzip -q -o VCTK-Corpus-0.92.zip
  cd ..
}

unpack_demand() {
  cd demand
  for z in *_48k.zip; do unzip -q -o "$z"; done
  cd ..
}

unpack_fsd() {
  cd fsd50k
  zip -q -s 0 FSD50K.dev_audio.zip --out joined.zip
  unzip -q -o joined.zip
  rm -f joined.zip
  cd ..
}

rirs() {
  python3 "$HERE/gen_rirs.py" near "$DATA/rir_near" --n 400 --seed 0
  python3 "$HERE/gen_rirs.py" far "$DATA/rir_far" --n 400 --seed 1
}

interferers() {
  for split in train valid test; do
    n=30; [ "$split" = train ] && n=60
    python3 "$HERE/make_interferers.py" \
      --speakers "$DATA/lists/spk_int_$split.txt" \
      --vctk-root "$DATA/vctk/wav48_silence_trimmed" \
      --rir-dir "$DATA/rir_far" \
      --outdir "$DATA/interferers_$split" \
      --per-speaker "$n" --seed 42
  done
}

hdf5() {
  build() {  # $1 type, $2 list, $3 output
    [ -s "hdf5/$3.hdf5" ] && return
    echo "  $3 <- $(wc -l < "lists/$2.txt") files"
    python3 "$PREP" --sr 48000 --mono --num_workers 16 "$1" "lists/$2.txt" "hdf5/$3.part.hdf5"
    mv "hdf5/$3.part.hdf5" "hdf5/$3.hdf5"
  }
  for s in TRAIN VALID TEST; do
    build speech "${s}_SPEECH"         "${s}_SET_SPEECH"
    build rir    "${s}_RIR"            "${s}_RIR"
    build noise  "${s}_NOISE_DEMAND"   "${s}_NOISE_DEMAND"
    build noise  "${s}_NOISE_FSD"      "${s}_NOISE_FSD"
    build noise  "${s}_NOISE_INTERFER" "${s}_NOISE_INTERFER"
  done
  du -sh hdf5
}

# ponytail: downloads run one after another (~1.5 h); fine overnight.
step dl_vctk       dl_vctk
step dl_demand     dl_demand
step dl_fsd        dl_fsd
step unpack_vctk   unpack_vctk
step unpack_demand unpack_demand
step unpack_fsd    unpack_fsd
step rirs          rirs
step split_vctk    python3 "$HERE/split_vctk.py" "$DATA"
step interferers   interferers
step lists         python3 "$HERE/build_lists.py" "$DATA"
step hdf5          hdf5
echo "### data ready  $(date '+%F %T')"
