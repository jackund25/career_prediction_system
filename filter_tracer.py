import pandas as pd
import os
import re

COLUMNS_TO_KEEP = [
    "NIM",
    "Program Studi",
    "Sebutkan kategori jenis pekerjaan yang Anda lakukan di tempat bekerja!",
    "Jelaskan tugas-tugas utama dalam pekerjaan Anda saat ini?",
    "Apakah pekerjaan yang Anda lakukan di tempat bekerja sesuai dengan bidang keilmuan?",
    "Seberapa erat hubungan bidang studi dengan pekerjaan Anda?",
    "Seberapa besar program studi Anda bermanfaat untuk hal-hal di bawah ini? [pembelajaran yang berkelanjutan dalam pekerjaan]",
    "Bidang usaha bekerja",
    "Organisasi apa yang paling aktif Anda ikuti selama menjalani perkuliahan? (nama organisasi)",
    "IP",
    "Sebutkan jenis kegiatan di organisasi yang aktif diikuti yang membantu mengasah kemampuan/skill Anda!",
    "Seberapa besar program studi Anda bermanfaat untuk hal-hal di bawah ini? [memulai pekerjaan]"
]

ALLOWED_JOB_STATUS = ["Bekerja", "Bekerja dan wiraswasta"]

def clean_dataframe(df):
    # Hapus semua karakter "-" di seluruh dataframe
    df = df.applymap(lambda x: re.sub(r'-', '', x).strip() if isinstance(x, str) else x)
    return df

def filter_and_export(input_path, output_path="filtered_output.csv"):
    file_ext = os.path.splitext(input_path)[1].lower()

    # Baca semua kolom sebagai teks
    if file_ext == ".xlsx":
        df = pd.read_excel(input_path, dtype=str)
    elif file_ext == ".csv":
        try:
            df = pd.read_csv(input_path, dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, dtype=str, encoding="latin-1")
    else:
        raise ValueError("Format file tidak didukung. Gunakan file Excel (.xlsx) atau CSV (.csv).")

    # Bersihkan semua tanda "-"
    df = clean_dataframe(df)

    # Filter pekerjaan
    if "Pekerjaan utama saat ini?" not in df.columns:
        raise KeyError("Kolom 'Pekerjaan utama saat ini?' tidak ditemukan di file input.")
    df = df[df["Pekerjaan utama saat ini?"].isin(ALLOWED_JOB_STATUS)]

    # Pilih kolom yang ada
    available_columns = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    missing_columns = [col for col in COLUMNS_TO_KEEP if col not in df.columns]
    if missing_columns:
        print("Peringatan: Kolom berikut tidak ditemukan di file:", missing_columns)

    filtered_df = df[available_columns]

    # Simpan ke CSV
    filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"File berhasil difilter, semua tanda '-' dihapus, dan disimpan ke: {output_path}")

if __name__ == "__main__":
    print("=== Filter Data Tracer Alumni ===")
    input_path = input("Masukkan path file input (Excel/CSV): ").strip('"')
    output_path = input("Masukkan path file output (default: filtered_output.csv): ").strip('"')
    if not output_path:
        output_path = "filtered_output.csv"

    try:
        filter_and_export(input_path, output_path)
    except Exception as e:
        print("Terjadi kesalahan:", e)
