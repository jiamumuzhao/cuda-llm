#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
def fail(s):print("VERIFY ERROR: "+s,file=sys.stderr);raise SystemExit(1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--package-dir",default="artifacts/phase15/qwen3-0.6b-f32");a=ap.parse_args();p=Path(a.package_dir)
 for x in ["model.meta","weights.bin","conversion_report.txt","tokenizer_fixtures.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json","tokenizer/generation_config.json"]:
  if not (p/x).exists():fail("missing "+str(p/x))
 try:
  import transformers
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(str(p/"tokenizer"),local_files_only=True)
 except Exception as e:fail("offline tokenizer load failed: "+repr(e))
 d=json.loads((p/"tokenizer_fixtures.json").read_text());
 for item in d["fixtures"]:
  rendered=tok.apply_chat_template(item["messages"],tokenize=False,add_generation_prompt=True) if item["kind"]=="chat" else item["rendered_prompt"]
  ids=tok.encode(rendered,add_special_tokens=item["kind"]!="chat")
  if rendered!=item["rendered_prompt"] or ids!=item["token_ids"]:fail("tokenizer fixture mismatch for "+item["kind"])
 print("verified package and",len(d["fixtures"]),"tokenizer fixtures")
if __name__=="__main__":main()
