# db.py - penyimpanan Postgres (Neon). Aktif bila env DATABASE_URL ada.
import os
import psycopg
from psycopg.types.json import Jsonb

URL = os.environ.get("DATABASE_URL")

def _c():  # prepare_threshold=None: wajib untuk connection string "pooled"
    return psycopg.connect(URL, autocommit=True, prepare_threshold=None)

def baca_semua():
    with _c() as c:
        return [r[0] for r in c.execute("select data from obat order by id")]

def tulis_semua(data):
    with _c() as c:
        for p in data:
            c.execute("insert into obat(id,data) values (%s,%s) on conflict (id) do update "
                      "set data=excluded.data, updated_at=now() where obat.data is distinct from excluded.data",
                      (p["id"], Jsonb(p)))
        c.execute("delete from obat where not (id = any(%s))", ([p["id"] for p in data],))

def versi():
    with _c() as c:
        return c.execute("select count(*)::text||':'||coalesce(max(updated_at)::text,'') from obat").fetchone()[0]

def foto_tambah(oid, jpg):
    with _c() as c:
        return c.execute("insert into foto(obat_id,jpg) values (%s,%s) returning id", (oid, jpg)).fetchone()[0]

def foto_ambil(fid):
    with _c() as c:
        r = c.execute("select jpg from foto where id=%s", (fid,)).fetchone()
    return bytes(r[0]) if r else None

def foto_hapus(fid):
    with _c() as c:
        c.execute("delete from foto where id=%s", (fid,))

def draft_simpan(tok, jpg):
    with _c() as c:
        c.execute("delete from draft where created_at < now() - interval '1 day'")
        c.execute("insert into draft(token,jpg) values (%s,%s)", (tok, jpg))

def draft_ambil(tok):
    with _c() as c:
        r = c.execute("select jpg from draft where token=%s", (tok,)).fetchone()
    return bytes(r[0]) if r else None