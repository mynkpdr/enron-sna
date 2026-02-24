import pandas as pd
import email
import os
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Regex to match any Enron-related internal domain
# This will match emails like:
# - employee+energy@sub.enron.com
# - test_user@hr.enron.com
# - x.y.z@services.enron.com
# - alpha-99@us.enron.com
ENRON_DOMAIN_RE = re.compile(r'[\w.+-]+@[\w.-]*enron\.com', re.I)

def process_chunk(chunk):
    """
    Worker function: Parses headers and filters for Enron domains.
    """
    parsed_rows = []
    header_fields = ['Date', 'From', 'To', 'Subject', 'X-From', 'X-To']
    
    for _, row in chunk.iterrows():
        msg = email.message_from_string(row['message'])
        
        sender = msg.get('From', '') or ''
        recipient = msg.get('To', '') or ''
        
        # Filtering logic: Only keep if sender or receiver is Enron-specific
        if ENRON_DOMAIN_RE.search(sender) or ENRON_DOMAIN_RE.search(recipient):
            data = {
                'file': row['file'],
                'body': msg.get_payload()
            }
            for field in header_fields:
                data[field] = msg.get(field)
            parsed_rows.append(data)
    
    # Return a DataFrame if we found matches, otherwise None
    return pd.DataFrame(parsed_rows) if parsed_rows else pd.DataFrame()

def fast_parse_filtered_enron(input_path, output_path):
    chunk_size = 15000  # Increased chunk size for better parallel efficiency
    
    # Using a context manager for the reader to handle file closing
    reader = pd.read_csv(input_path, chunksize=chunk_size)
    
    print("🚀 Starting Parallel Extraction (Enron Domains Only)...")
    
    with ProcessPoolExecutor() as executor:
        # Map the chunks to the workers
        futures = [executor.submit(process_chunk, chunk) for chunk in reader]
        
        first_chunk = True
        with tqdm(total=len(futures), desc="Processing Chunks") as pbar:
            for future in futures:
                df_result = future.result()
                
                if not df_result.empty:
                    mode = 'w' if first_chunk else 'a'
                    header = True if first_chunk else False
                    df_result.to_csv(output_path, index=False, mode=mode, header=header)
                    first_chunk = False
                
                pbar.update(1)

if __name__ == "__main__":
    INPUT = 'emails.csv'
    OUTPUT = 'enron_internal_only.csv'
    
    if os.path.exists(INPUT):
        fast_parse_filtered_enron(INPUT, OUTPUT)
        print(f"\n✅ Done! Filtered data saved to {OUTPUT}")
    else:
        print(f"❌ Error: {INPUT} not found.")