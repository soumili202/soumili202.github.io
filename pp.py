@app.post("/submit_button")
async def submit_button(
    user_query: str = Form(...),
    invoice_file: UploadFile = File(...),
    ledger_file: UploadFile = File(...)
):
    # 1) Parse invoice CSV
    invoice_content = await invoice_file.read()
    invoices = parse_invoice_csv(invoice_content.decode("utf-8"))
    write_json(INVOICE_PATH, invoices)

    # 2) Parse ledger CSV
    ledger_content = await ledger_file.read()
    ledgers = parse_ledger_csv(ledger_content.decode("utf-8"))
    write_json(LEDGER_PATH, ledgers)

    # 3) Save user query into integrator_meta.json
    write_json(META_PATH, {"user_query": user_query})

    # 4) Run aurix_engine.py automatically (same folder)
    try:
        subprocess.run(["python", "aurix_engine.py"], check=True)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed running AURIX Engine: {str(e)}"
        }

    return {
        "status": "success",
        "message": "Invoices, Ledgers, Query saved + final_prompt.txt generated.",
        "next_step": "Open final_prompt.txt → Paste into Copilot → Paste output into copilot_output.json"
    }