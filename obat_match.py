#!/usr/bin/env python3
# obat_match.py - scan kotak obat (uji lokal, satu file).
#   python obat_match.py web [--port 8000] [--https]   -> halaman kamera otomatis (http://localhost:8000)
#   python obat_match.py uji foto1.jpg foto2.jpg       -> uji satu/lebih foto dari terminal
#   python obat_match.py tambah f1.jpg f2.jpg [--merk X --nama "Amoxicillin" --kekuatan 500 --bentuk tablet --isi 10 --barcode 899...]
#   python obat_match.py daftar                        -> daftar katalog
#   (web) http://localhost:8000/tambah                 -> tambah obat / tambah foto / ubah / hapus (terisi otomatis dari foto)
# Katalog: folder katalog/ berisi katalog.json + foto kemasan. Tambah obat = tambah baris + foto, tanpa training.
#
# Cara kerja (urutan):
#   TAHAP 1  BARCODE : barcode dibaca dulu. Ada di katalog -> langsung "ok" (approve).
#                      Tidak ada di katalog / tidak terbaca -> lanjut tahap 2.
#   TAHAP 2  GAMBAR  : (a) SIFT + RANSAC mencocokkan desain kotak
#                      (b) OCR membaca MERK, NAMA OBAT, kekuatan, bentuk, isi
#                      (c) WARNA kotak (nama warna + histogram)
#                      Semua cocok -> "ok"; ada keraguan/konflik -> "ragu" (kuning, petugas memutuskan).
#   Setiap hasil membawa "rincian" (merk / nama / warna / kekuatan / bentuk / isi : terbaca vs katalog) supaya mudah dibaca.
import os, sys, json, re, time, threading, collections, difflib, uuid, shutil
import cv2, numpy as np
try:
    import zxingcpp                      # pembaca barcode (EAN-13, UPC, Code128, QR, DataMatrix, dll)
except ImportError:                      # tanpa pustaka ini tahap barcode dilewati, sistem tetap jalan lewat gambar
    zxingcpp = None

BASE = os.path.dirname(os.path.abspath(__file__))
KAT = os.environ.get("KATALOG", os.path.join(BASE, "katalog"))
MIN_INLIER = int(os.environ.get("MIN_INLIER", "14"))     # minimal titik cocok agar dianggap "mengenali desain"
RATIO_RAGU = float(os.environ.get("RATIO_RAGU", "0.6"))  # kandidat kedua >= 60% skor pertama -> mirip -> perlu OCR/petugas
WARNA_MAX = float(os.environ.get("WARNA_MAX", "0.50"))   # jarak warna maksimum (0 sama .. 1 beda)
WARNA_BEDA = float(os.environ.get("WARNA_BEDA", "0.15"))  # selisih jarak warna minimal agar warna dianggap pembeda dua kandidat
OCR_MIN_TEKS = int(os.environ.get("OCR_MIN_TEKS", "3"))  # minimal potongan teks terbaca agar merk/nama boleh dianggap "tidak cocok"
TEKS_MIN = float(os.environ.get("TEKS_MIN", "0.80"))     # kemiripan huruf minimal merk/nama (toleransi salah baca OCR)
BARCODE_ASING_RAGU = os.environ.get("BARCODE_ASING_RAGU", "0") == "1"  # 1 = barcode terbaca tapi tidak ada di katalog -> selalu "ragu"
MAXSIDE = 640
DRAFT = os.path.join(KAT, "_draft")

USE_DB = bool(os.environ.get("DATABASE_URL"))
if USE_DB:
    import db
_ver, _ver_at = None, 0.0

_lock = threading.Lock()
_sift = cv2.SIFT_create(nfeatures=1500)
_bf = cv2.BFMatcher()
_ocr = None
CAT = []
_BC = {}                                  # kode barcode (ternormalisasi) -> [entri katalog]

STOP = set("""TABLET TABLETS KAPLET KAPLETS CAPSULE CAPSULES KAPSUL MG MCG ML ISI STRIP BOX DUS TAB CAP OBAT SYRUP SIRUP
INJEKSI INJECTION SALEP KRIM CREAM OINTMENT PER EACH CONTAINS MENGANDUNG NETTO NET MERAH ORANYE KUNING HIJAU TOSCA BIRU
UNGU PINK NAVY TEAL RED BLUE GREEN YELLOW ORANGE PURPLE TUA PUTIH""".split())


def _resize(im, side=MAXSIDE):
    h, w = im.shape[:2]
    s = side / max(h, w)
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im


def _hist(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] > 60) & (hsv[..., 2] > 60)).astype(np.uint8) * 255
    if int(m.sum()) < 50 * 255:
        m = None
    h = cv2.calcHist([hsv], [0], m, [36], [0, 180]).flatten().astype("float32")
    return h / (h.sum() + 1e-9)


def _feats(im):
    g = cv2.createCLAHE(2.0, (8, 8)).apply(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return _sift.detectAndCompute(g, None)


def _inliers(q, r):
    kq, dq = q
    kr, dr = r
    if dq is None or dr is None or len(kq) < 8 or len(kr) < 8:
        return 0
    m = [a for a, b in _bf.knnMatch(dq, dr, k=2) if a.distance < 0.75 * b.distance]
    if len(m) < 8:
        return 0
    A = np.float32([kq[x.queryIdx].pt for x in m])
    B = np.float32([kr[x.trainIdx].pt for x in m])
    Hm, msk = cv2.findHomography(B, A, cv2.RANSAC, 4.0)
    return int(msk.sum()) if msk is not None else 0


# ----------------------------------------------------------------------------- tahap 1: barcode
def _kode_key(k):
    k = re.sub(r"\s+", "", str(k)).upper()
    return (k.lstrip("0") or "0") if k.isdigit() else k       # EAN-13 "0123..." == UPC-A "123..."


def _daftar_kode(x):
    if isinstance(x, (list, tuple)):
        x = ",".join(map(str, x))
    out = []
    for k in re.split(r"[,\s;]+", str(x or "")):
        if k and k not in out:
            out.append(k)
    return out


def baca_barcode(img):
    """BGR -> [{kode, format}]. Kosong bila tidak ada barcode / pustaka zxing-cpp belum terpasang."""
    if zxingcpp is None:
        return []
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    hasil = zxingcpp.read_barcodes(g)
    if not hasil and max(g.shape) < 1400:                       # barcode kecil -> coba diperbesar
        hasil = zxingcpp.read_barcodes(cv2.resize(g, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC))
    out, seen = [], set()
    for r in hasil:
        t = (r.text or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append({"kode": t, "format": str(getattr(r.format, "name", r.format))})
    return out


def _reindex():
    global _BC
    _BC = {}
    for c in CAT:
        for k in c.get("barcode") or []:
            _BC.setdefault(_kode_key(k), []).append(c)


# ----------------------------------------------------------------------------- warna
def _nama_warna(h, v):
    if h <= 7 or h >= 170: return "merah"
    if h <= 18: return "oranye"
    if h <= 35: return "kuning"
    if h <= 75: return "hijau"
    if h <= 98: return "tosca"
    if h <= 122: return "biru tua" if v < 125 else "biru"
    if h <= 152: return "ungu"
    return "pink"


def warna_info(im):
    """BGR -> {nama, hex}: warna dominan (yang berwarna/tidak putih-abu) pada kemasan."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > 70) & (hsv[..., 2] > 50)
    if int(m.sum()) < 0.01 * m.size:
        return {"nama": "putih/abu", "hex": "#d1d5db"}
    hh = hsv[..., 0]
    bins = np.bincount((hh[m] // 5).astype(int), minlength=36).astype(float)
    bins = bins + np.roll(bins, 1) + np.roll(bins, -1)          # sirkular (merah ada di 0 dan 175)
    c = int(np.argmax(bins)) * 5 + 2
    d = np.minimum(np.abs(hh.astype(int) - c), 180 - np.abs(hh.astype(int) - c))
    sel = m & (d <= 10)
    b, g, r = [int(x) for x in im[sel].mean(axis=0)]
    return {"nama": _nama_warna(c, float(hsv[..., 2][sel].mean())), "hex": f"#{r:02x}{g:02x}{b:02x}"}


# ----------------------------------------------------------------------------- OCR (merk, nama, kekuatan, bentuk, isi)
def _angka(x):
    try:
        v = float(str(x).replace(",", "."))
        return int(v) if v == int(v) else v
    except Exception:
        return None


def _kanon(t):                                  # samakan salah-baca OCR yang umum (0/O, 1/I, 5/S)
    return t.upper().translate(str.maketrans("015", "OIS"))


def ocr_baca(im):
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    r, _ = _ocr(im)
    items = []
    for box, teks, sc in (r or []):
        ys = [p[1] for p in box]
        items.append({"teks": teks.strip(), "h": float(max(ys) - min(ys))})
    t = " ".join(i["teks"] for i in items).upper()
    mg = [_angka(x) for x in re.findall(r"(\d{1,4}(?:[.,]\d+)?)\s*MG", t)]
    isi = [int(x) for x in re.findall(r"(\d{1,3})\s*(?:TABLET|CAPSULE|KAPLET|KAPSUL)", t)] or \
          [int(x) for x in re.findall(r"ISI\s*(\d{1,3})", t)]
    if "TABLET" in t or "KAPLET" in t: bentuk = "tablet"
    elif "CAPSULE" in t or "KAPSUL" in t: bentuk = "kapsul"
    elif "SYRUP" in t or "SIRUP" in t: bentuk = "sirup"
    elif "INJECTION" in t or "INJEKSI" in t: bentuk = "injeksi"
    elif re.search(r"OINTMENT|SALEP|CREAM|KRIM", t): bentuk = "salep"
    else: bentuk = None
    # saran merk & nama: teks beraksara dengan tinggi huruf terbesar yang bukan kata umum (tablet, mg, dst.)
    kand, sudah = [], set()
    for i in sorted(items, key=lambda x: -x["h"]):
        if re.search(r"\d|mg\s*$", i["teks"], re.I):             # lewati teks dosis/isi ("500 mg", "10 tablets", "soomg")
            continue
        kata = [k for k in re.findall(r"[A-Za-z]{3,}", i["teks"]) if k.upper() not in STOP]
        s = " ".join(kata)
        if kata and _kanon(s) not in sudah and len(s) <= 40:
            sudah.add(_kanon(s)); kand.append(s.title())
    return {"kekuatan": collections.Counter(mg).most_common(1)[0][0] if mg else None,
            "bentuk": bentuk, "isi": isi[0] if isi else None,
            "saran_merk": kand[0] if kand else None, "saran_nama": kand[1] if len(kand) > 1 else None,
            "_tok": [_kanon(x) for x in re.findall(r"[A-Za-z0-9]+", t)], "_n": len(items),
            "_teks": [i["teks"] for i in sorted(items, key=lambda x: -x["h"])[:6]]}


def ocr_fields(im):
    return {k: v for k, v in ocr_baca(im).items() if not k.startswith("_")}


def _cari_teks(target, tok):
    """Apakah kata-kata penting dari `target` (merk / nama) ada di tulisan kotak? -> None bila tak ada yang bisa dicek."""
    kata = [k for k in re.findall(r"[A-Za-z]+", target or "") if len(k) >= 3 and k.upper() not in STOP]
    if not kata:
        return None
    ok, ketemu = True, []
    for k in kata:
        kk, best = _kanon(k), (0.0, "")
        for t in tok:
            r = 1.0 if (len(kk) >= 4 and kk in t) else difflib.SequenceMatcher(None, kk, t).ratio()
            if r > best[0]:
                best = (r, t)
        ok &= best[0] >= TEKS_MIN
        if best[0] >= 0.5:
            ketemu.append(best[1])
    return {"ok": ok, "terbaca": " ".join(ketemu) or None}


def _beda(c, f):
    """Daftar alasan bila tulisan di kotak bertentangan dengan katalog."""
    b = []
    for k in ("kekuatan", "bentuk", "isi"):
        if f[k] is not None and c.get(k) is not None and f[k] != c[k]:
            b.append(f"{k} di kotak {f[k]} ≠ katalog {c[k]}")
    if f["_n"] >= OCR_MIN_TEKS:
        for k in ("merk", "nama"):
            r = _cari_teks(c.get(k), f["_tok"])
            if r and not r["ok"]:
                b.append(f"{k} katalog \"{c[k]}\" tidak ditemukan di tulisan kotak")
    return b


# ----------------------------------------------------------------------------- katalog
def _warna_entri(p, refs_im):
    w = p.get("warna")
    if isinstance(w, dict) and w.get("nama") and w.get("hex"):
        return w
    auto = warna_info(refs_im[0]) if refs_im else {"nama": "-", "hex": "#9ca3af"}
    return {"nama": w, "hex": auto["hex"]} if isinstance(w, str) and w else auto


def _buat_entri(p, pasang):
    """p: baris katalog.json; pasang: [(im, hist, feat)] -> entri memori."""
    return {**{k: v for k, v in p.items() if k not in ("foto", "warna")},
            "warna": _warna_entri(p, [x[0] for x in pasang]),
            "thumb": (p["foto"][0] if isinstance(p["foto"][0], str) else p["foto"][0]["file"]) if p.get("foto") else None,
            "refs": [{"hist": h, "feat": f} for _, h, f in pasang]}

def _baca_foto(fn):
    if fn.startswith("db:"):
        b = db.foto_ambil(int(fn[3:]))
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
    return cv2.imread(os.path.join(KAT, fn))

def _simpan_foto(oid, n, im):
    buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    if USE_DB:
        return "db:%d" % db.foto_tambah(oid, buf)
    fn = f"{oid}_{n}.jpg"
    while os.path.exists(os.path.join(KAT, fn)):
        fn = f"{oid}_{uuid.uuid4().hex[:4]}.jpg"
    open(os.path.join(KAT, fn), "wb").write(buf)
    return fn

def _hapus_file(fn):
    if fn.startswith("db:"):
        db.foto_hapus(int(fn[3:]))
    elif os.path.exists(os.path.join(KAT, fn)):
        os.remove(os.path.join(KAT, fn))

def _segarkan():            # sinkronkan antar-instance: muat ulang katalog bila database berubah (dicek tiap 10 detik)
    global _ver, _ver_at
    if not USE_DB or time.time() - _ver_at < 10:
        return
    _ver_at = time.time()
    v = db.versi()
    if v != _ver:
        _ver = v
        load_catalog()

def _catat_versi():         # dipanggil setelah menulis, agar instance ini tidak memuat ulang karena tulisannya sendiri
    global _ver, _ver_at
    if USE_DB:
        _ver, _ver_at = db.versi(), time.time()

def load_catalog():
    global CAT
    baru = []                                   # was: CAT = []
    for p in _baca_json():
        pasang = []
        for f in p["foto"]:
            fn, crop = (f, None) if isinstance(f, str) else (f["file"], f.get("crop"))
            im = _baca_foto(fn)
            if im is None:
                raise SystemExit(f"Foto katalog tidak terbaca: {fn}")
            if crop:
                im = im[crop[1]:crop[3], crop[0]:crop[2]]
            im = _resize(im)
            pasang.append((im, _hist(im), _feats(im)))
        baru.append(_buat_entri(p, pasang))
    CAT = baru
    _reindex()
    dobel = [k for k, v in _BC.items() if len(v) > 1]
    print(f"Katalog: {len(CAT)} obat, {sum(len(c['refs']) for c in CAT)} foto referensi, {len(_BC)} barcode"
          + (f"  PERINGATAN barcode ganda: {dobel}" if dobel else "")
          + ("" if zxingcpp else "  (zxing-cpp belum terpasang: tahap barcode dilewati)"))


def _pilihan(c, skor=None):
    return {"id": c["id"], "nama": c["nama"], "merk": c.get("merk"), "kekuatan": c.get("kekuatan"), "bentuk": c.get("bentuk"),
            "isi": c.get("isi"), "warna": c["warna"], "ukuran_kotak": c.get("ukuran_kotak"), "skor": skor}


def _baris(k, label, terbaca, katalog, cocok, **x):
    return {"k": k, "label": label, "terbaca": terbaca, "katalog": katalog, "cocok": cocok, **x}


def _bandingkan(a, b):
    return None if a is None or b is None else (a == b)


def match(img, barcode=True):
    """img BGR -> dict(status ok|ragu|tidak, metode barcode|gambar, pilihan, kandidat, rincian, barcode, ocr, alasan, ms)"""
    _segarkan()
    t0 = time.time()
    out = {"status": "tidak", "metode": None, "pilihan": None, "kandidat": [], "ocr": None, "alasan": [],
           "rincian": [], "barcode": None, "warna_terbaca": None}
    # ---- TAHAP 1: barcode
    kode_asing = None
    if barcode:
        for b in baca_barcode(img):
            with _lock:
                ents = list(_BC.get(_kode_key(b["kode"]), []))
            if len(ents) == 1:
                c = ents[0]
                out.update(status="ok", metode="barcode", pilihan=_pilihan(c), barcode={"kode": b["kode"], "ada": True},
                           kandidat=[{"id": c["id"], "nama": c["nama"], "skor": None}],
                           rincian=[_baris("barcode", "Barcode", b["kode"], ", ".join(c.get("barcode") or []), True),
                                    _baris("merk", "Merk", None, c.get("merk"), None),
                                    _baris("nama", "Nama obat", None, c["nama"], None),
                                    _baris("warna", "Warna", None, c["warna"]["nama"], None, hex_katalog=c["warna"]["hex"]),
                                    _baris("kekuatan", "Kekuatan", None, c.get("kekuatan"), None),
                                    _baris("bentuk", "Bentuk", None, c.get("bentuk"), None),
                                    _baris("isi", "Isi", None, c.get("isi"), None)])
                out["ms"] = int((time.time() - t0) * 1000)
                return out
            if len(ents) > 1:                              # satu barcode dipakai beberapa entri -> petugas yang memilih
                out.update(status="ragu", metode="barcode", pilihan=_pilihan(ents[0]), barcode={"kode": b["kode"], "ada": True},
                           kandidat=[{"id": c["id"], "nama": c["nama"], "skor": None} for c in ents],
                           alasan=["barcode ini terdaftar pada beberapa obat: " + ", ".join(c["nama"] for c in ents)])
                out["ms"] = int((time.time() - t0) * 1000)
                return out
            kode_asing = kode_asing or b["kode"]
        if kode_asing:
            out["barcode"] = {"kode": kode_asing, "ada": False}
    # ---- TAHAP 2: gambar (desain + tulisan + warna)
    with _lock:
        q = _resize(img)
        qf, qh = _feats(q), _hist(q)
        sk = sorted(((max(_inliers(qf, r["feat"]) for r in c["refs"]), c) for c in CAT), key=lambda x: -x[0])
        cands = [(s, c) for s, c in sk if s >= MIN_INLIER]
        if not cands:
            out["kandidat"] = [{"id": c["id"], "nama": c["nama"], "skor": s} for s, c in sk[:2] if s >= 6]
            out["ms"] = int((time.time() - t0) * 1000)
            return out
        top = cands[0][0]
        group = [(s, c) for s, c in cands if s >= RATIO_RAGU * top]
        f = ocr_baca(q)
        wq = warna_info(q)
        alasan = []
        ok_ = [(s, c) for s, c in group if not _beda(c, f)]
        if len(ok_) == 1:
            pilih = ok_[0]
        elif len(ok_) == 0:
            pilih = group[0]
            alasan += _beda(pilih[1], f)
        else:
            # desain & tulisan sama-sama cocok: bila warnanya jelas membedakan, pilih yang warnanya paling dekat
            wds = sorted(((min(cv2.compareHist(qh, r["hist"], cv2.HISTCMP_BHATTACHARYYA) for r in c["refs"]), s, c) for s, c in ok_),
                         key=lambda x: x[0])
            if wds[0][0] <= WARNA_MAX and wds[1][0] - wds[0][0] >= WARNA_BEDA:
                pilih = (wds[0][1], wds[0][2])
            else:
                pilih = ok_[0]
                alasan.append("mirip dengan obat lain: " + ", ".join(c["nama"] for s, c in ok_[1:]))
        c = pilih[1]
        wd = min(cv2.compareHist(qh, r["hist"], cv2.HISTCMP_BHATTACHARYYA) for r in c["refs"])
        if wd > WARNA_MAX:
            alasan.append(f"warna kotak berbeda dari katalog (terbaca {wq['nama']}, katalog {c['warna']['nama']})")
        if kode_asing and BARCODE_ASING_RAGU:
            alasan.append(f"barcode {kode_asing} belum terdaftar di katalog")
        rm, rn = _cari_teks(c.get("merk"), f["_tok"]), _cari_teks(c.get("nama"), f["_tok"])
        teks_ok = lambda r: None if r is None else (r["ok"] if r["terbaca"] or f["_n"] >= OCR_MIN_TEKS else None)
        out["rincian"] = [
            _baris("barcode", "Barcode", kode_asing, ", ".join(c.get("barcode") or []) or None, None) if kode_asing else None,
            _baris("merk", "Merk", (rm or {}).get("terbaca"), c.get("merk"), teks_ok(rm)),
            _baris("nama", "Nama obat", (rn or {}).get("terbaca"), c["nama"], teks_ok(rn)),
            _baris("warna", "Warna", wq["nama"], c["warna"]["nama"], wd <= WARNA_MAX, hex_terbaca=wq["hex"], hex_katalog=c["warna"]["hex"]),
            _baris("kekuatan", "Kekuatan", f["kekuatan"], c.get("kekuatan"), _bandingkan(f["kekuatan"], c.get("kekuatan"))),
            _baris("bentuk", "Bentuk", f["bentuk"], c.get("bentuk"), _bandingkan(f["bentuk"], c.get("bentuk"))),
            _baris("isi", "Isi", f["isi"], c.get("isi"), _bandingkan(f["isi"], c.get("isi")))]
        out["rincian"] = [x for x in out["rincian"] if x]
        out.update(status="ok" if not alasan else "ragu", metode="gambar", alasan=alasan, pilihan=_pilihan(c, pilih[0]),
                   kandidat=[{"id": cc["id"], "nama": cc["nama"], "skor": s} for s, cc in cands[:3]],
                   ocr={"kekuatan": f["kekuatan"], "bentuk": f["bentuk"], "isi": f["isi"], "teks": f["_teks"]},
                   warna_terbaca=wq)
        out["ms"] = int((time.time() - t0) * 1000)
        return out


# ----------------------------------------------------------------------------- generator & pengelola katalog
_kat_lock = threading.Lock()


def _baca_json():
    if USE_DB: return db.baca_semua()        # di _baca_json
    if USE_DB: return db.tulis_semua(data)   # di _tulis_json
    return json.load(open(os.path.join(KAT, "katalog.json"), encoding="utf-8"))


def _tulis_json(data):
    if USE_DB: return db.baca_semua()        # di _baca_json
    if USE_DB: return db.tulis_semua(data)   # di _tulis_json
    path = os.path.join(KAT, "katalog.json")
    if os.path.exists(path):
        shutil.copyfile(path, path + ".bak")                  # cadangan versi sebelumnya
    tmp = path + ".tmp"
    json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _cek_barcode_unik(data, kodes, kecuali=None):
    for k in kodes:
        for p in data:
            if p["id"] != kecuali and _kode_key(k) in [_kode_key(x) for x in p.get("barcode", [])]:
                raise ValueError(f"barcode {k} sudah dipakai {p['id']} {p['nama']}")


def _sinkron(entry):
    """Perbarui data teks entri di memori (tanpa menghitung ulang foto)."""
    cur = next((c for c in CAT if c["id"] == entry["id"]), None)
    if cur:
        for k in ("nama", "merk", "kekuatan", "bentuk", "isi", "barcode", "ukuran_kotak"):
            if k in entry: cur[k] = entry[k]
            else: cur.pop(k, None)
        if isinstance(entry.get("warna"), dict): cur["warna"] = entry["warna"]
    _reindex()


def tambah_obat(fotos, meta, obat_id=None):
    """fotos: list bytes JPEG/PNG. meta: nama, merk, kekuatan, bentuk, isi, barcode, ukuran_kotak, warna.
    obat_id diisi -> MENAMBAH FOTO (dan barcode bila diberikan) ke obat itu. Return dict(id, peringatan[], dicek[])."""
    imgs = []
    for b in fotos:
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("ada foto yang tidak terbaca")
        imgs.append(_resize(im, 1000))
    if not imgs:
        raise ValueError("minimal 1 foto")
    kodes = _daftar_kode(meta.get("barcode"))
    with _kat_lock:
        data = _baca_json()
        if obat_id:
            entry = next((p for p in data if p["id"] == obat_id), None)
            if not entry:
                raise ValueError(f"obat {obat_id} tidak ada")
            _cek_barcode_unik(data, kodes, kecuali=obat_id)
            for k in kodes:
                if k not in entry.setdefault("barcode", []):
                    entry["barcode"].append(k)
        else:
            merk = (meta.get("merk") or "").strip()
            bentuk = (meta.get("bentuk") or "").strip().lower()
            kek, isi = _angka(meta.get("kekuatan")), _angka(meta.get("isi"))
            nama = (meta.get("nama") or "").strip() or " ".join(x for x in (merk, bentuk, f"{kek} mg" if kek else "") if x)
            if not nama or not bentuk:
                raise ValueError("nama (atau merk) dan bentuk wajib diisi")
            for p in data:
                if p["nama"].lower() == nama.lower() and p.get("kekuatan") == kek and p.get("bentuk") == bentuk and p.get("isi") == isi \
                        and (p.get("merk") or "").lower() == merk.lower():
                    raise ValueError(f"sudah ada: {p['id']} {p['nama']} (pakai 'tambah foto' ke obat itu)")
            _cek_barcode_unik(data, kodes)
            nums = [int(m.group(1)) for p in data if (m := re.match(r"OBT-(\d+)", p["id"]))]
            entry = {"id": f"OBT-{max(nums, default=0) + 1:02d}", "nama": nama, "kekuatan": kek, "bentuk": bentuk, "isi": isi, "foto": []}
            if merk: entry["merk"] = merk
            if kodes: entry["barcode"] = kodes
            if (meta.get("ukuran_kotak") or "").strip(): entry["ukuran_kotak"] = meta["ukuran_kotak"].strip()
            if (meta.get("warna") or "").strip(): entry["warna"] = meta["warna"].strip()
            data.append(entry)
        pasang = []
        for im in imgs:
            fn = _simpan_foto(entry["id"], len(entry["foto"]) + 1, im)
            entry["foto"].append(fn)
            pasang.append((_resize(im), _hist(_resize(im)), _feats(_resize(im))))
        w = warna_info(pasang[0][0])
        if not isinstance(entry.get("warna"), dict):          # simpan nama + hex warna agar bisa dilihat/diubah petugas
            entry["warna"] = {"nama": entry.get("warna") or w["nama"], "hex": w["hex"]}
        _tulis_json(data)
        _catat_versi()
        with _lock:                                           # perbarui katalog di memori tanpa restart
            cur = next((c for c in CAT if c["id"] == entry["id"]), None)
            if cur:
                cur["refs"].extend({"hist": h, "feat": f} for _, h, f in pasang)
                _sinkron(entry)
            else:
                CAT.append(_buat_entri(entry, pasang))
                _reindex()
    # uji diri (hanya lewat gambar, barcode tidak dipakai): tiap foto baru harus dikenali sebagai obat ini
    per, dicek = [], []
    for im in imgs:
        r = match(im, barcode=False)
        got = (r["pilihan"] or {}).get("id")
        dicek.append({"status": r["status"], "dikenali_sebagai": got})
        if got != entry["id"] or r["status"] != "ok":
            per.append("foto ini belum dikenali jelas sebagai obat ini (" + ("; ".join(r["alasan"]) or r["status"]) + ") - ulangi foto lebih terang/dekat")
        lain = [k for k in r["kandidat"] if k["id"] != entry["id"] and r["pilihan"] and k["skor"] >= RATIO_RAGU * r["pilihan"]["skor"]]
        if lain:
            per.append("desain mirip dengan " + ", ".join(k["nama"] for k in lain) + " - tulisan kekuatan/isi harus terbaca jelas agar tidak tertukar")
    if not entry.get("barcode"):
        per.append("belum ada barcode - tambahkan agar scan berikutnya langsung approve lewat barcode")
    return {"id": entry["id"], "nama": entry["nama"], "jumlah_foto": len(entry["foto"]), "barcode": entry.get("barcode", []),
            "peringatan": sorted(set(per)), "dicek": dicek}


def ubah_obat(obat_id, meta):
    """Ubah data teks obat (hanya kunci yang dikirim). barcode = daftar dipisah koma (menggantikan yang lama)."""
    with _kat_lock:
        data = _baca_json()
        e = next((p for p in data if p["id"] == obat_id), None)
        if not e:
            raise ValueError(f"obat {obat_id} tidak ada")
        for k in ("nama", "merk", "bentuk", "ukuran_kotak"):
            if k in meta:
                v = (meta[k] or "").strip()
                v = v.lower() if k == "bentuk" else v
                if not v and k in ("nama", "bentuk"):
                    raise ValueError(f"{k} tidak boleh kosong")
                if v: e[k] = v
                else: e.pop(k, None)
        for k in ("kekuatan", "isi"):
            if k in meta:
                v = _angka(meta[k])
                if v is None and str(meta[k] or "").strip():
                    raise ValueError(f"{k} harus angka")
                if v is None: e[k] = None
                else: e[k] = v
        if "barcode" in meta:
            kodes = _daftar_kode(meta["barcode"])
            _cek_barcode_unik(data, kodes, kecuali=obat_id)
            if kodes: e["barcode"] = kodes
            else: e.pop("barcode", None)
        if "warna" in meta and (meta["warna"] or "").strip():
            w = e.get("warna") if isinstance(e.get("warna"), dict) else {}
            e["warna"] = {"nama": meta["warna"].strip(), "hex": w.get("hex") or next((c["warna"]["hex"] for c in CAT if c["id"] == obat_id), "#9ca3af")}
        _tulis_json(data)
        _catat_versi()
        with _lock:
            _sinkron(e)
    return {"id": e["id"], "nama": e["nama"]}


def hapus_obat(obat_id):
    with _kat_lock:
        data = _baca_json()
        e = next((p for p in data if p["id"] == obat_id), None)
        if not e:
            raise ValueError(f"obat {obat_id} tidak ada")
        sisa = [p for p in data if p["id"] != obat_id]
        _tulis_json(sisa)
        _catat_versi()                                    # katalog.json.bak menyimpan versi sebelum dihapus
        pakai = {(f if isinstance(f, str) else f["file"]) for p in sisa for f in p["foto"]}
        for f in e["foto"]:
            fn = f if isinstance(f, str) else f["file"]
            if fn not in pakai: 
                _hapus_file(fn)
        with _lock:
            CAT[:] = [c for c in CAT if c["id"] != obat_id]
            _reindex()
    return {"id": obat_id}


def daftar_katalog():
    _segarkan()
    return [{"id": c["id"], "nama": c["nama"], "merk": c.get("merk"), "kekuatan": c.get("kekuatan"), "bentuk": c.get("bentuk"),
             "isi": c.get("isi"), "barcode": c.get("barcode", []), "warna": c["warna"], "ukuran_kotak": c.get("ukuran_kotak"),
             "thumb": c.get("thumb"), "n_foto": len(c["refs"])} for c in CAT]


def analisa_foto(img):
    """Untuk halaman tambah: baca otomatis barcode, merk, nama, kekuatan, bentuk, isi, warna + saran kualitas foto."""
    q = _resize(img, 800)
    with _lock:
        f = ocr_baca(q)
    g = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
    tips = []
    if cv2.Laplacian(g, cv2.CV_64F).var() < 60: tips.append("foto agak buram - tahan HP stabil / dekatkan sampai tulisan tajam")
    if g.mean() < 70: tips.append("foto terlalu gelap - cari tempat lebih terang")
    if g.mean() > 215: tips.append("foto terlalu terang/silau - hindari pantulan lampu")
    if f["_n"] < OCR_MIN_TEKS: tips.append("tulisan di kemasan hampir tidak terbaca - penuhi bingkai dengan kotak, tulisan menghadap kamera")
    return {**{k: v for k, v in f.items() if not k.startswith("_")}, "warna": warna_info(q),
            "barcode": [b["kode"] for b in baca_barcode(img)], "tips": tips}


def simpan_draft(b):
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("gambar tidak terbaca")
    tok = uuid.uuid4().hex[:12]
    data = cv2.imencode(".jpg", _resize(im, 1000), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    if USE_DB:
        db.draft_simpan(tok, data)
        return tok
    os.makedirs(DRAFT, exist_ok=True)
    open(os.path.join(DRAFT, tok + ".jpg"), "wb").write(data)
    return tok


def baca_draft(tok):
    if not re.fullmatch(r"[0-9a-f]{12}", tok or ""):
        return None
    if USE_DB:
        return db.draft_ambil(tok)
    p = os.path.join(DRAFT, tok + ".jpg")
    return open(p, "rb").read() if os.path.exists(p) else None


# ----------------------------------------------------------------------------- web lokal
def make_app():
    from flask import Flask, request, jsonify, Response, send_from_directory
    app = Flask(__name__)

    def _pin():
        p = os.environ.get("ADMIN_PIN", "")
        return (not p) or request.headers.get("X-Pin") == p

    def _gambar():
        return cv2.imdecode(np.frombuffer(request.get_data(), np.uint8), cv2.IMREAD_COLOR)

    @app.get("/")
    def index():
        return Response(PAGE.replace("__N__", str(len(CAT))), mimetype="text/html")

    @app.post("/api/match")
    def api_match():

        img = _gambar()
        if img is None:
            return jsonify(error="gambar tidak terbaca"), 400
        return jsonify(match(img))

    @app.get("/tambah")
    def tambah_page():
        return Response(PAGE_TAMBAH, mimetype="text/html")

    @app.get("/api/katalog")
    def api_katalog():
        return jsonify(daftar_katalog())

    @app.get("/foto/<path:fn>")
    def foto(fn):
        if fn.startswith("db:"):
            b = db.foto_ambil(int(fn[3:]))
            return Response(b, mimetype="image/jpeg") if b else ("", 404)
        return send_from_directory(KAT, os.path.basename(fn))

    @app.post("/api/baca")                      # baca otomatis dari foto: barcode, merk, nama, kekuatan, bentuk, isi, warna
    def api_baca():
        img = _gambar()
        if img is None:
            return jsonify(error="gambar tidak terbaca"), 400
        return jsonify(analisa_foto(img))

    @app.post("/api/draft")                     # simpan frame dari halaman scan (untuk "perbaiki / daftarkan")
    def api_draft():
        try:
            return jsonify(draft=simpan_draft(request.get_data()))
        except ValueError as e:
            return jsonify(error=str(e)), 400

    @app.get("/api/draft/<tok>")
    def api_draft_get(tok):
        b = baca_draft(tok)
        return Response(b, mimetype="image/jpeg") if b else (jsonify(error="draf tidak ada"), 404)

    @app.post("/api/tambah")
    def api_tambah():
        if not _pin():
            return jsonify(error="PIN salah"), 401
        try:
            fotos = [f.read() for f in request.files.getlist("foto")]
            fotos += [b for b in (baca_draft(t) for t in request.form.getlist("draft")) if b]
            return jsonify(tambah_obat(fotos, request.form.to_dict(), request.form.get("id") or None))
        except ValueError as e:
            return jsonify(error=str(e)), 400

    @app.post("/api/ubah/<oid>")
    def api_ubah(oid):
        if not _pin():
            return jsonify(error="PIN salah"), 401
        try:
            return jsonify(ubah_obat(oid, request.get_json(force=True) or {}))
        except ValueError as e:
            return jsonify(error=str(e)), 400

    @app.post("/api/hapus/<oid>")
    def api_hapus(oid):
        if not _pin():
            return jsonify(error="PIN salah"), 401
        try:
            return jsonify(hapus_obat(oid))
        except ValueError as e:
            return jsonify(error=str(e)), 400
    return app


PAGE = r"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"><title>Scan Kotak Obat</title>
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#0b0f14;color:#fff;font-family:system-ui,sans-serif}
#app{display:flex;flex-direction:column;height:100dvh}#top{padding:8px 16px;display:flex;justify-content:space-between;font-size:15px}
#cam{position:relative;flex:1;min-height:160px;background:#000;overflow:hidden}video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}
#ring{position:absolute;inset:6%;border:3px dashed rgba(255,255,255,.7);border-radius:18px;transition:border-color .15s}
#hint{position:absolute;left:0;right:0;top:8px;text-align:center;font-size:13px;text-shadow:0 1px 3px #000}
#banner{position:absolute;left:10px;right:10px;bottom:8px;padding:10px;border-radius:14px;text-align:center;display:none}
#banner b{font-size:21px;display:block}#banner small{display:block;font-size:13px;margin-top:2px;opacity:.95}
.ok{background:#16a34a}.ragu{background:#d97706}.tidak{background:#374151}
#start{position:absolute;inset:0;background:#0b0f14;z-index:5;display:flex;flex-direction:column;gap:14px;align-items:center;justify-content:center;text-align:center;padding:24px}
button{font:inherit;border:0;border-radius:12px;padding:14px 22px;background:#2563eb;color:#fff;font-weight:700;font-size:18px}
button.sec{background:#1f2937;font-size:15px;padding:10px 14px}
#panel{background:#111827;padding:10px 14px;max-height:46dvh;overflow:auto;display:none}
.tag{display:inline-block;font-size:12px;font-weight:800;letter-spacing:.5px;padding:3px 9px;border-radius:99px;margin-right:8px}
.t-barcode{background:#1d4ed8}.t-gambar{background:#7c3aed}
#phead{font-size:15px;margin-bottom:6px;display:flex;align-items:center;flex-wrap:wrap;gap:4px}
table{width:100%;border-collapse:collapse;font-size:16px}th{text-align:left;color:#9aa4b2;font-weight:600;width:27%;padding:5px 4px;font-size:14px}
td{padding:5px 4px;border-top:1px solid #1f2937}td.m{width:28px;text-align:center;font-weight:800;font-size:20px}
thead th{font-size:12px;border:0}.ctrue .m{color:#4ade80}.cfalse .m{color:#f87171}.cfalse td{color:#fca5a5}.cnull .m{color:#6b7280}
.sw{display:inline-block;width:16px;height:16px;border-radius:50%;vertical-align:-3px;margin-right:6px;border:1px solid #fff5}
#alasan{margin:8px 0 0;color:#fbbf24;font-size:14px}#acts{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
#sheet{position:fixed;inset:0;background:#000a;z-index:9;display:none;align-items:flex-end}
#sheet>div{background:#111827;width:100%;padding:16px;border-radius:18px 18px 0 0;max-height:80dvh;overflow:auto}
select,input{width:100%;padding:12px;border-radius:10px;border:1px solid #374151;background:#0b0f14;color:#fff;font-size:16px;margin:8px 0}
#bot{background:#0b0f14;padding:6px 14px;font-size:12px;color:#9aa4b2;max-height:70px;overflow:auto}#bot div{padding:1px 0}
.h-ok{color:#4ade80}.h-ragu{color:#fbbf24}</style></head><body><div id="app">
<div id="top"><div>Scan Kotak Obat · <a href="/tambah" style="color:#93c5fd">+ tambah / ubah obat</a></div><div id="ms">katalog __N__ obat</div></div>
<div id="cam"><video id="v" playsinline muted></video><div id="ring"></div><div id="hint">Arahkan BARCODE dulu · bila tak ada, penuhi bingkai dengan kotak obat</div><div id="banner"></div>
<div id="start"><div style="font-size:24px;font-weight:800">Scan Kotak Obat</div><div style="color:#9aa4b2;font-size:14px">1) barcode &rarr; 2) bila tak ada: merk, nama, warna &amp; ukuran dari gambar<br>Kamera dan suara akan diaktifkan</div><button id="go">Mulai</button><div id="msg" style="color:#f87171;font-size:14px"></div></div></div>
<div id="panel"><div id="phead"></div><table><thead><tr><th></th><th>Terbaca</th><th>Di katalog</th><th></th></tr></thead><tbody id="rows"></tbody></table>
<div id="alasan"></div><div id="acts"></div></div>
<div id="bot"></div></div>
<div id="sheet"><div><b style="font-size:18px">Perbaiki / daftarkan obat ini</b>
<div style="color:#9aa4b2;font-size:14px;margin-top:4px">Foto yang sedang terlihat dipakai sebagai foto katalog<span id="kinfo"></span>.</div>
<select id="pilih"></select><button id="simpan" style="width:100%">Simpan ke obat terpilih</button>
<button class="sec" id="baru" style="width:100%;margin-top:8px">Daftarkan sebagai obat BARU</button>
<button class="sec" id="batal" style="width:100%;margin-top:8px;background:#374151">Batal</button><div id="sres" style="margin-top:8px;font-size:14px"></div></div></div>
<script>
const $=id=>document.getElementById(id);let AC,busy=false,pause=0,last='',lastAt=0,miss=0,curBlob=null,curKode=null;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function beep(f,d,at=0){try{const o=AC.createOscillator(),g=AC.createGain();o.frequency.value=f;o.connect(g);g.connect(AC.destination);g.gain.value=.25;o.start(AC.currentTime+at);o.stop(AC.currentTime+at+d);}catch(e){}}
function say(t,l){try{speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(t);u.lang=l;u.rate=1.15;speechSynthesis.speak(u);}catch(e){}}
function show(cls,title,sub,col){const b=$('banner');b.className=cls;b.innerHTML='<b>'+esc(title)+'</b>'+(sub?'<small>'+esc(sub)+'</small>':'');b.style.display='block';$('ring').style.borderColor=col;
 clearTimeout(show.t);show.t=setTimeout(()=>{b.style.display='none';$('ring').style.borderColor='rgba(255,255,255,.7)';},2200);}
function hist(cls,t){const d=document.createElement('div');d.className=cls;d.textContent=new Date().toTimeString().slice(0,8)+'  '+t;$('bot').prepend(d);while($('bot').children.length>25)$('bot').lastChild.remove();}
const IC={true:'✓',false:'✗',null:'–'};
const sw=h=>h?'<span class="sw" style="background:'+esc(h)+'"></span>':'';
function panel(r){
  const p=r.pilihan,bc=r.barcode;
  $('phead').innerHTML='<span class="tag t-'+r.metode+'">'+(r.metode==='barcode'?'BARCODE':'GAMBAR')+'</span><b>'+esc(p.merk?p.merk+' · ':'')+esc(p.nama)+'</b>';
  $('rows').innerHTML=(r.rincian||[]).map(x=>{let t=x.terbaca,k=x.katalog;
    if(x.k==='warna'){t=sw(x.hex_terbaca)+esc(t??'—');k=sw(x.hex_katalog)+esc(k??'—');}
    else if(x.k==='kekuatan'){t=t==null?'—':esc(t)+' mg';k=k==null?'—':esc(k)+' mg';}
    else{t=esc(t??'—');k=esc(k??'—');}
    return '<tr class="c'+x.cocok+'"><th>'+esc(x.label)+'</th><td>'+t+'</td><td>'+k+'</td><td class="m">'+IC[x.cocok]+'</td></tr>';}).join('');
  $('alasan').innerHTML=(r.alasan||[]).map(a=>'⚠ '+esc(a)).join('<br>')+((bc&&!bc.ada)?'<br>ℹ Barcode '+esc(bc.kode)+' belum terdaftar — dicek lewat gambar.':'');
  acts(r.status!=='ok'||(bc&&!bc.ada));$('panel').style.display='block';}
function acts(show_){$('acts').innerHTML=show_?'<button class="sec" id="fix">Perbaiki / daftarkan obat ini</button>':'';if(show_)$('fix').onclick=openSheet;}
async function api(url,opt){let r=await fetch(url,opt);if(r.status===401){const p=prompt('PIN admin');if(p){localStorage.setItem('apin',p);opt.headers={...(opt.headers||{}),'X-Pin':p};r=await fetch(url,opt);}}return r;}
async function openSheet(){if(!curBlob)return;$('sheet').style.display='flex';$('sres').textContent='';$('kinfo').textContent=curKode?' dan barcode '+curKode+' dihubungkan ke obat':'';
  const K=await (await fetch('/api/katalog')).json();$('pilih').innerHTML=K.map(k=>'<option value="'+esc(k.id)+'">'+esc(k.id+' · '+(k.merk?k.merk+' ':'')+k.nama)+'</option>').join('');}
$('batal').onclick=()=>$('sheet').style.display='none';
async function draft(){const r=await (await fetch('/api/draft',{method:'POST',body:curBlob})).json();return r.draft;}
$('baru').onclick=async()=>{const d=await draft();location.href='/tambah?draft='+d+(curKode?'&kode='+encodeURIComponent(curKode):'');};
$('simpan').onclick=async()=>{$('sres').textContent='Menyimpan…';const d=await draft();const fd=new FormData();fd.append('id',$('pilih').value);fd.append('draft',d);if(curKode)fd.append('barcode',curKode);
  const r=await api('/api/tambah',{method:'POST',body:fd,headers:{'X-Pin':localStorage.getItem('apin')||''}});const j=await r.json();
  $('sres').innerHTML=j.error?'<span style="color:#f87171">'+esc(j.error)+'</span>':'✓ Tersimpan ke <b>'+esc(j.id)+'</b> ('+j.jumlah_foto+' foto'+(j.barcode&&j.barcode.length?', barcode '+esc(j.barcode.join(', ')):'')+')'+(j.peringatan.length?'<br>⚠ '+j.peringatan.map(esc).join('<br>⚠ '):'');
  if(!j.error){last='';setTimeout(()=>$('sheet').style.display='none',1800);}};
const cv=document.createElement('canvas'),cx=cv.getContext('2d');
async function tick(){
  if(busy||Date.now()<pause||$('sheet').style.display==='flex'){setTimeout(tick,120);return;}
  const v=$('v');if(!v.videoWidth){setTimeout(tick,200);return;}
  busy=true;
  try{
    const s=Math.min(1,960/Math.max(v.videoWidth,v.videoHeight));cv.width=v.videoWidth*s;cv.height=v.videoHeight*s;cx.drawImage(v,0,0,cv.width,cv.height);
    const blob=await new Promise(r=>cv.toBlob(r,'image/jpeg',.85));curBlob=blob;
    const t0=performance.now();const r=await (await fetch('/api/match',{method:'POST',body:blob})).json();const dt=Math.round(performance.now()-t0);
    $('ms').textContent='katalog __N__ obat · '+dt+' ms';curKode=r.barcode&&!r.barcode.ada?r.barcode.kode:null;
    const key=r.pilihan?r.pilihan.id+r.status:'';
    if(r.status==='tidak'){miss++;if(miss===4){show('tidak','Mencari…','Dekatkan barcode / penuhi bingkai dengan kotak','#9aa4b2');
        $('phead').innerHTML='<b>Tidak dikenali</b>';$('rows').innerHTML='';$('alasan').textContent=curKode?'Barcode '+curKode+' belum terdaftar.':'Belum ada obat di katalog yang cocok.';acts(true);$('panel').style.display='block';}}
    else{miss=0;const now=Date.now();const p=r.pilihan;panel(r);
      if(!(key===last&&now-lastAt<3500)){last=key;lastAt=now;const via=r.metode==='barcode'?'barcode':'gambar';
        if(r.status==='ok'){show('ok','✓ '+p.nama,(p.merk?p.merk+' · ':'')+p.id+' · '+(p.kekuatan??'-')+' mg · '+p.bentuk+' · isi '+p.isi+' · via '+via,'#4ade80');beep(880,.12);say('approve','en-US');hist('h-ok','ok  '+via+'  '+p.id+'  '+p.nama+'  ('+dt+' ms)');navigator.vibrate&&navigator.vibrate(60);}
        else{show('ragu','CEK MANUAL: '+p.nama,(r.alasan||[]).join(' · '),'#fbbf24');beep(330,.18);beep(330,.18,.25);say('cek manual','id-ID');hist('h-ragu','ragu  '+via+'  '+p.id+'  '+(r.alasan||[]).join('; '));navigator.vibrate&&navigator.vibrate([80,60,80]);}
        pause=Date.now()+1300;}}
  }catch(e){}
  busy=false;setTimeout(tick,60);
}
$('go').onclick=async()=>{try{AC=new (window.AudioContext||window.webkitAudioContext)();try{speechSynthesis.speak(new SpeechSynthesisUtterance(''));}catch(e){}
  const st=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}}});
  const v=$('v');v.srcObject=st;await v.play();$('start').style.display='none';try{navigator.wakeLock&&navigator.wakeLock.request('screen').catch(()=>{});}catch(e){}tick();
 }catch(e){$('msg').textContent=(e&&e.message)||String(e);}};
</script></body></html>"""


PAGE_TAMBAH = r"""<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tambah / Ubah Obat</title>
<style>*{box-sizing:border-box}body{margin:0;background:#0b0f14;color:#fff;font-family:system-ui,sans-serif;padding:16px;max-width:600px;margin:auto}
h1{font-size:21px;margin:6px 0 10px}label{display:block;font-size:14px;color:#9aa4b2;margin:12px 0 4px}label i{font-style:normal;color:#4ade80;font-size:12px;margin-left:6px}
input,select{width:100%;padding:13px;border-radius:10px;border:1px solid #374151;background:#111827;color:#fff;font-size:18px}input.auto{border-color:#16a34a;background:#052e16}
button{font:inherit;border:0;border-radius:12px;padding:15px;background:#2563eb;color:#fff;font-weight:700;font-size:18px;width:100%;margin-top:14px}
button.sec{background:#1f2937;font-size:15px;padding:11px;margin-top:8px}button.red{background:#7f1d1d;font-size:15px;padding:11px}.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
#tabs{display:flex;gap:8px;margin:8px 0 6px}#tabs button{margin:0;background:#1f2937;font-size:16px}#tabs button.on{background:#2563eb}
#thumbs{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap}#thumbs img{height:90px;border-radius:8px}
.box{margin-top:12px;padding:12px;border-radius:12px;font-size:15px;line-height:1.5}.ok{background:#14532d}.err{background:#7f1d1d}.warn{background:#78350f}.info{background:#1e3a5f}
.sw{display:inline-block;width:22px;height:22px;border-radius:50%;vertical-align:middle;border:1px solid #fff5;margin-right:8px}
.card{background:#111827;border-radius:12px;padding:10px;margin-top:10px}.card .h{display:flex;gap:10px;align-items:center}.card img{width:48px;height:60px;object-fit:cover;border-radius:6px}
.card small{color:#9aa4b2}.bc{font-size:12px;color:#93c5fd}a{color:#93c5fd}</style></head><body>
<a href="/">&larr; kembali ke scan</a><h1>Tambah / ubah obat</h1>
<div id="tabs"><button id="t1" class="on">Tambah</button><button id="t2">Daftar &amp; ubah</button></div>

<div id="v1">
<label>Mau apa?</label><select id="id"><option value="">Obat BARU</option></select>
<label>1. Foto kemasan (1-3 foto: tegak penuh bingkai, terang, tanpa silau)</label>
<input id="fo" type="file" accept="image/*" capture="environment" multiple><div id="thumbs"></div><div id="tips"></div>
<label>2. Foto barcode dari dekat (opsional - lebih akurat)</label><input id="fb" type="file" accept="image/*" capture="environment">
<div class="box info" id="bacainfo" style="display:none"></div>
<div id="form">
<label>Merk <i id="a_merk"></i></label><input id="merk" placeholder="mis. Kimia Farma">
<label>Nama obat <i id="a_nama"></i></label><input id="nama" placeholder="mis. Amoxicillin (boleh kosong: dibuat otomatis)">
<div class="row"><div><label>Kekuatan (mg) <i id="a_kek"></i></label><input id="kek" inputmode="decimal"></div><div><label>Isi per kemasan <i id="a_isi"></i></label><input id="isi" inputmode="numeric"></div></div>
<div class="row"><div><label>Bentuk <i id="a_bentuk"></i></label><select id="bentuk"><option>tablet</option><option>kapsul</option><option>sirup</option><option>injeksi</option><option>salep</option><option>lainnya</option></select></div>
<div><label>Warna kemasan <i id="a_warna"></i></label><input id="warna" placeholder="otomatis dari foto"></div></div>
<div id="swrow" style="margin-top:6px;color:#9aa4b2;font-size:13px"></div>
<label>Ukuran kotak (opsional, mis. 9 x 6 x 2 cm)</label><input id="ukk"></div>
<label>Barcode (pisahkan koma) <i id="a_bc"></i></label><input id="bc" inputmode="numeric" placeholder="otomatis bila terbaca">
<button id="simpan">Simpan ke katalog</button><div class="box" id="out" style="display:none"></div></div>

<div id="v2" style="display:none"><input id="cari" placeholder="Cari nama / merk / barcode..." style="margin-top:8px"><div id="list"></div></div>
<script>
const $=i=>document.getElementById(i);const Q=new URLSearchParams(location.search);let KAT=[],FILES=[],WARNA_HEX='';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function out(c,t){const o=$('out');o.className='box '+c;o.innerHTML=t;o.style.display='block';}
async function api(url,opt){opt.headers={...(opt.headers||{}),'X-Pin':localStorage.getItem('apin')||''};let r=await fetch(url,opt);
  if(r.status===401){const p=prompt('PIN admin');if(p){localStorage.setItem('apin',p);opt.headers['X-Pin']=p;r=await fetch(url,opt);}}return r;}
function tab(n){$('v1').style.display=n==1?'block':'none';$('v2').style.display=n==2?'block':'none';$('t1').className=n==1?'on':'';$('t2').className=n==2?'on':'';if(n==2)daftar();}
$('t1').onclick=()=>tab(1);$('t2').onclick=()=>tab(2);
async function muat(){KAT=await (await fetch('/api/katalog')).json();
  $('id').innerHTML='<option value="">Obat BARU</option>'+KAT.map(k=>`<option value="${esc(k.id)}">Tambah foto/barcode ke: ${esc(k.id)} ${esc((k.merk?k.merk+' ':'')+k.nama)} (${k.n_foto} foto)</option>`).join('');
  if(Q.get('id'))$('id').value=Q.get('id');$('id').onchange();}
$('id').onchange=()=>{$('form').style.display=$('id').value?'none':'block';};
function thumbs(){$('thumbs').innerHTML='';FILES.forEach(f=>{const i=new Image();i.src=URL.createObjectURL(f);$('thumbs').appendChild(i);});}
function isi(id,val,aid){const e=$(id);if(val==null||val===''||e.value&&e.dataset.manual)return;e.value=val;e.classList.add('auto');$(aid).textContent='terisi otomatis - cek';}
['merk','nama','kek','isi','bentuk','warna','bc'].forEach(k=>$(k).addEventListener('input',e=>{e.target.dataset.manual=1;e.target.classList.remove('auto');}));
async function analisa(files){
  if(!files.length)return;const b=$('bacainfo');b.style.display='block';b.textContent='Membaca foto…';let tips=[],got=false;
  for(const f of files.slice(0,3)){
    const r=await (await fetch('/api/baca',{method:'POST',body:f})).json();if(r.error)continue;
    if(!got&&(r.saran_merk||r.kekuatan||r.bentuk)){got=true;isi('merk',r.saran_merk,'a_merk');isi('nama',r.saran_nama,'a_nama');isi('kek',r.kekuatan,'a_kek');isi('isi',r.isi,'a_isi');if(r.bentuk)isi('bentuk',r.bentuk,'a_bentuk');
      isi('warna',r.warna.nama,'a_warna');WARNA_HEX=r.warna.hex;$('swrow').innerHTML='<span class="sw" style="background:'+esc(r.warna.hex)+'"></span>warna dominan terbaca dari foto';tips=r.tips||[];}
    if(r.barcode&&r.barcode.length&&!$('bc').dataset.manual){$('bc').value=r.barcode.join(', ');$('bc').classList.add('auto');$('a_bc').textContent='terbaca dari foto';}
  }
  b.innerHTML=got?'Terbaca dari foto. Isian hijau = otomatis, <b>cek lalu ubah bila perlu</b>.':'Tulisan belum terbaca - isi manual atau ulangi foto lebih dekat/terang.';
  $('tips').innerHTML=tips.length?'<div class="box warn">⚠ '+tips.map(esc).join('<br>⚠ ')+'</div>':'';}
$('fo').onchange=async()=>{FILES=[...$('fo').files].slice(0,3);thumbs();await analisa(FILES);};
$('fb').onchange=async()=>{const f=await kecil($('fb').files[0],1600);if(!f)return;const r=await (await fetch('/api/baca',{method:'POST',body:f})).json();
  if(r.barcode&&r.barcode.length){$('bc').value=r.barcode.join(', ');$('bc').classList.add('auto');$('a_bc').textContent='terbaca dari foto barcode';out('ok','Barcode terbaca: <b>'+esc(r.barcode.join(', '))+'</b>');}else out('warn','Barcode belum terbaca - dekatkan, terang, garis barcode lurus. Atau ketik manual.');};
$('simpan').onclick=async()=>{if(!FILES.length)return out('err','Pilih foto kemasan dulu');
  const fd=new FormData();FILES.forEach(f=>fd.append('foto',f));fd.append('barcode',$('bc').value);
  if($('id').value)fd.append('id',$('id').value);else{['merk','nama','kek','isi','bentuk','warna'].forEach(k=>fd.append(k=='kek'?'kekuatan':k,$(k).value));fd.append('ukuran_kotak',$('ukk').value);}
  out('warn','Menyimpan…');const r=await api('/api/tambah',{method:'POST',body:fd});const j=await r.json();if(j.error)return out('err',esc(j.error));
  out(j.peringatan.length?'warn':'ok','✓ Tersimpan: <b>'+esc(j.id)+'</b> '+esc(j.nama)+' ('+j.jumlah_foto+' foto'+(j.barcode.length?', barcode '+esc(j.barcode.join(', ')):'')+')'+(j.peringatan.length?'<br>⚠ '+j.peringatan.map(esc).join('<br>⚠ '):'<br>Foto dikenali dengan benar.'));
  FILES=[];$('fo').value='';$('thumbs').innerHTML='';muat();};
// ---- daftar & ubah
function daftar(){const q=$('cari').value.toLowerCase();
  $('list').innerHTML=KAT.filter(k=>(k.id+k.nama+(k.merk||'')+(k.barcode||[]).join(' ')).toLowerCase().includes(q)).map(k=>`<div class="card" id="c_${esc(k.id)}"><div class="h"><img src="/foto/${esc(k.thumb)}"><div style="flex:1"><b>${esc((k.merk?k.merk+' · ':'')+k.nama)}</b><br><small>${esc(k.id)} · ${k.kekuatan??'-'} mg · ${esc(k.bentuk)} · isi ${k.isi??'-'} · <span class="sw" style="width:11px;height:11px;margin:0 3px 0 0;background:${esc(k.warna.hex)}"></span>${esc(k.warna.nama)}</small><br><span class="bc">${k.barcode.length?'barcode: '+esc(k.barcode.join(', ')):'⚠ belum ada barcode'}</span></div></div>
  <button class="sec" onclick="ubah('${esc(k.id)}')">Ubah data</button> <button class="sec" onclick="location.href='/tambah?id=${esc(k.id)}'">+ foto / barcode</button><div id="e_${esc(k.id)}"></div></div>`).join('')||'<div class="box info">Tidak ada yang cocok.</div>';}
$('cari').oninput=daftar;
function ubah(id){const k=KAT.find(x=>x.id==id),e=$('e_'+id);if(e.innerHTML){e.innerHTML='';return;}
  const f=(n,l,v)=>`<label>${l}</label><input id="${n}_${id}" value="${esc(v)}">`;
  e.innerHTML=f('merk','Merk',k.merk)+f('nama','Nama obat',k.nama)+`<div class="row"><div>${f('kek','Kekuatan (mg)',k.kekuatan)}</div><div>${f('isi','Isi',k.isi)}</div></div><div class="row"><div>${f('bentuk','Bentuk',k.bentuk)}</div><div>${f('warna','Warna',k.warna.nama)}</div></div>`+f('ukk','Ukuran kotak',k.ukuran_kotak)+f('bc','Barcode (pisahkan koma)',(k.barcode||[]).join(', '))+
   `<button onclick="simpanUbah('${id}')">Simpan perubahan</button><button class="red" onclick="hapus('${id}')">Hapus obat ini</button><div class="box" id="r_${id}" style="display:none"></div>`;}
async function kecil(f,max=1000){const b=await createImageBitmap(f);const s=Math.min(1,max/Math.max(b.width,b.height));
  const c=document.createElement('canvas');c.width=b.width*s;c.height=b.height*s;c.getContext('2d').drawImage(b,0,0,c.width,c.height);
  return new Promise(r=>c.toBlob(x=>r(new File([x],'foto.jpg',{type:'image/jpeg'})),'image/jpeg',.85));}
$('fo').onchange=async()=>{FILES=await Promise.all([...$('fo').files].slice(0,3).map(f=>kecil(f)));thumbs();await analisa(FILES);};
async function simpanUbah(id){const g=n=>$(n+'_'+id).value;const body={merk:g('merk'),nama:g('nama'),kekuatan:g('kek'),isi:g('isi'),bentuk:g('bentuk'),warna:g('warna'),ukuran_kotak:g('ukk'),barcode:g('bc')};
  const r=await api('/api/ubah/'+id,{method:'POST',body:JSON.stringify(body),headers:{'Content-Type':'application/json'}});const j=await r.json();const o=$('r_'+id);o.style.display='block';
  if(j.error){o.className='box err';o.textContent=j.error;return;}o.className='box ok';o.textContent='✓ Tersimpan';await muat();daftar();}
async function hapus(id){if(!confirm('Hapus '+id+' dari katalog? (katalog.json.bak menyimpan cadangan)'))return;const r=await api('/api/hapus/'+id,{method:'POST'});const j=await r.json();if(j.error)return alert(j.error);await muat();daftar();}
// ---- terima draf dari halaman scan (foto + barcode yang baru terbaca)
(async()=>{await muat();
  if(Q.get('kode')){$('bc').value=Q.get('kode');$('bc').classList.add('auto');$('a_bc').textContent='dari hasil scan';}
  if(Q.get('draft')){const b=await (await fetch('/api/draft/'+Q.get('draft'))).blob();FILES=[new File([b],'scan.jpg',{type:'image/jpeg'})];thumbs();await analisa(FILES);}})();
</script></body></html>"""


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__ or "pakai: web | uji foto.jpg")
    load_catalog()
    if a[0] == "uji":
        for fn in a[1:]:
            r = match(cv2.imread(fn))
            print(fn, "->", json.dumps(r, ensure_ascii=False))
    elif a[0] == "daftar":
        for c in daftar_katalog():
            print(f"{c['id']:8} {(c['merk'] or '-'):14} {c['nama']:36} {c['kekuatan']} mg {c['bentuk']} isi {c['isi']}  warna {c['warna']['nama']}  barcode {','.join(c['barcode']) or '-'}")
    elif a[0] == "tambah":
        def opt(k):
            return a[a.index(k) + 1] if k in a else None
        fl = [x for x in a[1:] if os.path.isfile(x)]
        meta = {k: opt("--" + k) for k in ("merk", "nama", "kekuatan", "bentuk", "isi", "barcode", "ukuran_kotak")}
        if not fl:
            raise SystemExit("sebutkan minimal 1 file foto")
        im0 = cv2.imread(fl[0])
        if not opt("--id"):
            f = ocr_fields(_resize(im0))
            print("Terbaca dari foto:", json.dumps(f, ensure_ascii=False))
            meta["merk"] = meta["merk"] or f["saran_merk"]
            meta["nama"] = meta["nama"] or f["saran_nama"]
            for k in ("kekuatan", "bentuk", "isi"):
                meta[k] = meta[k] or (str(f[k]) if f[k] is not None else None)
        if not meta["barcode"]:
            kodes = [b["kode"] for fn in fl for b in baca_barcode(cv2.imread(fn))]
            if kodes:
                meta["barcode"] = ",".join(kodes)
                print("Barcode terbaca dari foto:", meta["barcode"])
        print(json.dumps(tambah_obat([open(x, "rb").read() for x in fl], meta, opt("--id")), ensure_ascii=False, indent=1))
    elif a[0] == "migrasi":      # salin katalog.json + foto lokal ke database (jalankan SEKALI)
        data = json.load(open(os.path.join(KAT, "katalog.json"), encoding="utf-8"))
        for p in data:
            baru = []
            for f in p["foto"]:
                fn, crop = (f, None) if isinstance(f, str) else (f["file"], f.get("crop"))
                fid = "db:%d" % db.foto_tambah(p["id"], open(os.path.join(KAT, fn), "rb").read())
                baru.append(fid if not crop else {"file": fid, "crop": crop})
            p["foto"] = baru
        db.tulis_semua(data)
        print("Dipindahkan:", len(data), "obat")
    elif a[0] == "web":
        port = int(a[a.index("--port") + 1]) if "--port" in a else 8000
        ctx = "adhoc" if "--https" in a else None
        print(f"Buka: {'https' if ctx else 'http'}://localhost:{port}   (dari HP: pakai IP laptop, wajib --https)")
        make_app().run(host="0.0.0.0", port=port, threaded=True, ssl_context=ctx)
