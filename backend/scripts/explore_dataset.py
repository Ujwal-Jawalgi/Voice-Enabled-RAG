import pyarrow.parquet as pq
import fsspec

def main():
    print("Fetching exact target_lang codes using fsspec and pyarrow...")
    
    fs = fsspec.filesystem("hf")
    
    # Read first row of Hindi
    with fs.open("datasets/ai4bharat/MSMARCO-XI/validation/hinval.parquet", "rb") as f:
        # read_table can take just 1 row
        pf = pq.ParquetFile(f)
        batch = next(pf.iter_batches(batch_size=1, columns=["target_lang"]))
        hin_code = batch["target_lang"][0].as_py()
        print(f"Hindi target_lang code: {hin_code}")
        
    # Read first row of Kannada
    with fs.open("datasets/ai4bharat/MSMARCO-XI/validation/kanval.parquet", "rb") as f:
        pf = pq.ParquetFile(f)
        batch = next(pf.iter_batches(batch_size=1, columns=["target_lang"]))
        kan_code = batch["target_lang"][0].as_py()
        print(f"Kannada target_lang code: {kan_code}")

if __name__ == "__main__":
    main()
