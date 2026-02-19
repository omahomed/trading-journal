# ==============================================================================
# PAGE 10: TRADE MANAGER (FULL CONTEXT: BUY/SELL NOTES & RULES)
# ==============================================================================
elif page == "Trade Manager":
    st.header(f"TRADE MANAGER ({CURR_PORT_NAME})")
    
    # Initialize files if missing
    if not os.path.exists(DETAILS_FILE): 
        pd.DataFrame(columns=['Trade_ID','Ticker','Action','Date','Shares','Amount','Value','Rule','Notes','Realized_PL','Stop_Loss','Trx_ID']).to_csv(DETAILS_FILE, index=False)
    if not os.path.exists(SUMMARY_FILE): 
        pd.DataFrame(columns=['Trade_ID','Ticker','Status','Open_Date','Total_Shares','Avg_Entry','Avg_Exit','Total_Cost','Realized_PL','Unrealized_PL','Rule','Notes','Buy_Notes','Sell_Rule','Sell_Notes']).to_csv(SUMMARY_FILE, index=False)
    
    df_d = load_data(DETAILS_FILE)
    df_s = load_data(SUMMARY_FILE)
    
    # --- SCHEMA UPDATE: Add Risk_Budget ---
    if 'Risk_Budget' not in df_s.columns:
        df_s['Risk_Budget'] = 0.0 # Initialize with 0
        # Optional: Backfill logic could go here, but 0 is safer for now.

    
    # --- SCHEMA FIXES ---
    # 1. Rename legacy 'Buy_Rule' -> 'Rule'
    if 'Buy_Rule' in df_s.columns and 'Rule' not in df_s.columns:
        df_s.rename(columns={'Buy_Rule': 'Rule'}, inplace=True)
    if 'Rule' not in df_s.columns: df_s['Rule'] = ""

    # 2. Ensure new dedicated Note/Rule columns exist
    for col in ['Buy_Notes', 'Sell_Rule', 'Sell_Notes']:
        if col not in df_s.columns: df_s[col] = ""

    valid_sum_cols = ['Trade_ID', 'Ticker', 'Status', 'Open_Date', 'Shares', 'Avg_Entry', 'Total_Cost', 'Unrealized_PL', 'Return_Pct', 'Rule', 'Buy_Notes', 'Sell_Rule']
    valid_sum_cols = [c for c in valid_sum_cols if c in df_s.columns]

    # Updated Tab List (CY placed before All Campaigns)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab_cy, tab10, tab11 = st.tabs([
    "Log Buy", 
    "Log Sell", 
    "Update Prices", 
    "Edit Transaction", 
    "Database Health", 
    "Delete Trade", 
    "Active Campaign Summary", 
    "Active Campaign Detailed", 
    "Detailed Trade Log", 
    "CY Campaigns (2026)",  # <--- Moved Here (Position 10)
    "All Campaigns",        # <--- Pushed to Position 11
    "Performance Audit"
])

# --- TAB 1: LOG BUY ---
    with tab1:
        st.caption("Live Entry Calculator")
        
        # Session State Init
        if 'b_tick' not in st.session_state: st.session_state['b_tick'] = ""
        if 'b_id' not in st.session_state: st.session_state['b_id'] = ""
        if 'b_shs' not in st.session_state: st.session_state['b_shs'] = 0
        if 'b_px' not in st.session_state: st.session_state['b_px'] = 0.0
        if 'b_note' not in st.session_state: st.session_state['b_note'] = ""
        if 'b_trx' not in st.session_state: st.session_state['b_trx'] = ""
        if 'b_sl_pct' not in st.session_state: st.session_state['b_sl_pct'] = 8.0
        if 'b_stop_val' not in st.session_state: st.session_state['b_stop_val'] = 0.0

        c_top1, c_top2 = st.columns(2)
        trade_type = c_top1.radio("Action Type", ["Start New Campaign", "Scale In (Add to Existing)"], horizontal=True)
        
        now = datetime.now()
        b_date = c_top2.date_input("Date", now, key="b_date_input")
        b_time = c_top2.time_input("Time", now.time(), step=60, key="b_time_input")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        # --- 1. TICKER & STRATEGY SELECTION ---
        if trade_type == "Start New Campaign":
            b_tick = c1.text_input("Ticker Symbol", key="b_tick")
            if b_tick: b_tick = b_tick.upper() 
            
            now_ym = datetime.now().strftime("%Y%m") 
            default_id = f"{now_ym}-001"
            if not df_s.empty:
                relevant_ids = [str(x) for x in df_s['Trade_ID'] if str(x).startswith(now_ym)]
                if relevant_ids:
                    try:
                        last_seq = max([int(x.split('-')[-1]) for x in relevant_ids if '-' in x])
                        new_seq = last_seq + 1
                        default_id = f"{now_ym}-{new_seq:03d}"
                    except: pass
            if st.session_state['b_id'] == "": st.session_state['b_id'] = default_id  
            b_id = c2.text_input("Trade ID", key="b_id")
            b_rule = st.selectbox("Buy Rule", BUY_RULES)
        else:
            # Scale In Logic
            open_opts = df_s[df_s['Status']=='OPEN'].copy()
            b_tick, b_id = "", ""
            if not open_opts.empty:
                open_opts = open_opts.sort_values('Ticker')
                opts = ["Select..."] + [f"{r['Ticker']} | {r['Trade_ID']}" for _, r in open_opts.iterrows()]
                sel_camp = c1.selectbox("Select Existing Campaign", opts, key="b_scale_sel")
                if sel_camp and sel_camp != "Select...":
                    b_tick, b_id = sel_camp.split(" | ")
                    curr_row = open_opts[open_opts['Trade_ID']==b_id].iloc[0]
                    c2.info(f"Holding: {int(curr_row['Shares'])} shs @ ${curr_row['Avg_Entry']:.2f}")
            else: c1.warning("No Open Campaigns.")
            b_rule = st.selectbox("Add Rule", BUY_RULES)

        # --- 2. RISK BUDGET CALCULATOR (New Campaigns) ---
        risk_budget_dol = 0.0
        if trade_type == "Start New Campaign":
            st.markdown("#### 💰 Risk Budgeting")
            def_equity = 100000.0
            if os.path.exists(JOURNAL_FILE):
                 try: 
                     j_df = pd.read_csv(JOURNAL_FILE)
                     if not j_df.empty:
                        val_str = str(j_df['End NLV'].iloc[0]).replace('$','').replace(',','')
                        def_equity = float(val_str)
                 except: pass
            
            rb1, rb2, rb3 = st.columns(3)
            risk_pct_input = rb1.number_input("Risk % of Equity", value=0.50, step=0.05, format="%.2f")
            risk_budget_dol = def_equity * (risk_pct_input / 100)
            
            rb2.metric("Account Equity (Prev)", f"${def_equity:,.2f}")
            rb3.metric("Hard Risk Budget ($)", f"${risk_budget_dol:.2f}")

        # --- 3. EXECUTION DETAILS ---
        c3, c4 = st.columns(2)
        b_shs = c3.number_input("Shares", min_value=0, step=1, key="b_shs")
        b_px = c4.number_input("Entry Price ($)", min_value=0.0, step=0.1, format="%.2f", key="b_px")
        
        # --- RBM STOP CALCULATION (THE GUARDRAIL) ---
        rbm_stop = 0.0
        if trade_type == "Start New Campaign" and risk_budget_dol > 0 and b_shs > 0:
            # RBM Stop = Price - (Budget / Shares)
            # Example: 150 - (500 / 100) = 145
            risk_per_share_allowable = risk_budget_dol / b_shs
            rbm_stop = b_px - risk_per_share_allowable
            
            # Show the Guardrail
            st.info(f"🛑 **RBM Stop (Hard Deck):** ${rbm_stop:.2f} (To maintain ${risk_budget_dol:.0f} risk)")

        st.markdown("#### 🛡️ Risk Management")
        c_stop1, c_stop2 = st.columns(2)
        with c_stop1: stop_mode = st.radio("Stop Loss Mode", ["Price Level ($)", "Percentage (%)"], horizontal=True)
        with c_stop2:
            if stop_mode == "Percentage (%)":
                sl_pct = st.number_input("Stop Loss %", value=8.0, step=0.5, format="%.1f", key="b_sl_pct")
                b_stop = b_px * (1 - (sl_pct/100)) if b_px > 0 else 0.0
                st.metric("Calculated Stop", f"${b_stop:.2f}", delta=f"-{sl_pct}%")
            else:
                def_val = float(b_px * 0.92) if (st.session_state['b_stop_val'] == 0.0 and b_px > 0) else st.session_state['b_stop_val']
                b_stop = st.number_input("Stop Price ($)", min_value=0.0, step=0.1, value=def_val, format="%.2f", key="b_stop_val")
                if b_px > 0 and b_stop > 0: 
                    actual_pct = ((b_px - b_stop) / b_px) * 100
                    st.caption(f"Implied Risk: {actual_pct:.2f}%")
        
        # --- VALIDATION MESSAGE ---
        if trade_type == "Start New Campaign" and rbm_stop > 0:
            # Check if user stop is below the RBM stop (Violation)
            # Assuming Long trade: Stop must be >= RBM Stop to be safe
            if b_stop < rbm_stop:
                excess_risk = (rbm_stop - b_stop) * b_shs
                st.error(f"⚠️ **RISK VIOLATION:** Your stop (${b_stop:.2f}) is too wide! It exceeds budget by ${excess_risk:.2f}.")
            elif b_stop >= b_px:
                st.warning("⚠️ Stop Price is above Entry Price.")
            else:
                st.success(f"✅ **WITHIN BUDGET:** Your stop respects the Risk Limit (Above ${rbm_stop:.2f}).")

        st.markdown("---")
        c_note1, c_note2 = st.columns(2)
        b_note = c_note1.text_input("Buy Rationale (Notes)", key="b_note")
        b_trx = c_note2.text_input("Manual Trx ID (Optional)", key="b_trx")

        if st.button("LOG BUY ORDER", type="primary", use_container_width=True):
            if b_tick and b_id and b_shs > 0 and b_px > 0:
                ts = datetime.combine(b_date, b_time).strftime("%Y-%m-%d %H:%M")
                cost = b_shs * b_px
                if not b_trx: b_trx = generate_trx_id(df_d, b_id, 'BUY', ts)
                
                # Save Detail
                new_d = {'Trade_ID': b_id, 'Trx_ID': b_trx, 'Ticker': b_tick, 'Action': 'BUY', 'Date': ts, 'Shares': b_shs, 'Amount': b_px, 'Value': cost, 'Rule': b_rule, 'Notes': b_note, 'Realized_PL': 0, 'Stop_Loss': b_stop}
                df_d = pd.concat([df_d, pd.DataFrame([new_d])], ignore_index=True)
                
                if trade_type == "Start New Campaign":
                    # --- NEW: SAVE 'Risk_Budget' & 'Buy_Notes' ---
                    new_s = {
                        'Trade_ID': b_id, 'Ticker': b_tick, 'Status': 'OPEN', 'Open_Date': ts, 
                        'Shares': 0, 'Avg_Entry': 0, 'Total_Cost': 0, 'Realized_PL': 0, 'Unrealized_PL': 0, 
                        'Rule': b_rule, 
                        'Notes': b_note,       
                        'Buy_Notes': b_note,   
                        'Risk_Budget': risk_budget_dol, # <--- Saved Here
                        'Sell_Rule': '', 'Sell_Notes': ''
                    }
                    df_s = pd.concat([df_s, pd.DataFrame([new_s])], ignore_index=True)
                
                secure_save(df_d, DETAILS_FILE)
                df_d, df_s = update_campaign_summary(b_id, df_d, df_s) # Syncs math
                secure_save(df_d, DETAILS_FILE)
                secure_save(df_s, SUMMARY_FILE)
                
                st.success(f"✅ EXECUTED: Bought {b_shs} {b_tick} @ ${b_px}")
                for k in ['b_tick','b_id','b_shs','b_px','b_note','b_trx','b_stop_val']:
                    if k in st.session_state: del st.session_state[k]
                st.rerun()
            else: st.error("⚠️ Missing Data.")

    # --- TAB 2: LOG SELL ---
    with tab2:
        open_opts = df_s[df_s['Status']=='OPEN'].copy()
        if not open_opts.empty:
             open_opts = open_opts.sort_values('Ticker')
             s_opts = [f"{r['Ticker']} | {r['Trade_ID']}" for _, r in open_opts.iterrows()]
             sel_sell = st.selectbox("Select Trade to Sell", s_opts)
             if sel_sell:
                 s_tick, s_id = sel_sell.split(" | ")
                 row = open_opts[open_opts['Trade_ID']==s_id].iloc[0]
                 st.info(f"Selling {s_tick} (Own {int(row['Shares'])} shs)")
                 
                 c1, c2 = st.columns(2)
                 s_date = c1.date_input("Date", datetime.now(), key='s_date')
                 s_time = c2.time_input("Time", datetime.now().time(), step=60, key='s_time')
                 
                 c3, c4 = st.columns(2)
                 s_shs = c3.number_input("Shares", min_value=1, max_value=int(row['Shares']), step=1)
                 s_px = c4.number_input("Price", min_value=0.0, step=0.1)
                 
                 # --- NEW: EXPLICIT SELL RULE & NOTES ---
                 c5, c6 = st.columns(2)
                 s_rule = c5.selectbox("Sell Rule / Reason", SELL_RULES)
                 s_note = c6.text_input("Sell Context / Notes", key='s_note', placeholder="Why did you sell?")
                 s_trx = st.text_input("Manual Trx ID (Optional)", key='s_trx')
                 
                 if st.button("LOG SELL ORDER", type="primary"):
                    ts = datetime.combine(s_date, s_time).strftime("%Y-%m-%d %H:%M")
                    proc = s_shs * s_px
                    if not s_trx: s_trx = generate_trx_id(df_d, s_id, 'SELL', ts)
                    
                    # Log Detail (Rule = Sell Rule, Notes = Sell Note)
                    new_d = {'Trade_ID':s_id, 'Trx_ID': s_trx, 'Ticker':s_tick, 'Action':'SELL', 'Date':ts, 'Shares':s_shs, 'Amount':s_px, 'Value':proc, 'Rule':s_rule, 'Notes': s_note, 'Realized_PL': 0}
                    df_d = pd.concat([df_d, pd.DataFrame([new_d])], ignore_index=True)
                    secure_save(df_d, DETAILS_FILE)
                    
                    # Sync Math
                    df_d, df_s = update_campaign_summary(s_id, df_d, df_s)
                    
                    # --- CRITICAL FIX: FORCE WRITE SELL DATA TO SUMMARY ---
                    # We do this AFTER update_campaign_summary to ensure it doesn't get overwritten
                    idx = df_s[df_s['Trade_ID'] == s_id].index
                    if not idx.empty:
                        df_s.at[idx[0], 'Sell_Rule'] = s_rule
                        # Append or Overwrite notes? Overwrite is cleaner for "Last Action Reason"
                        df_s.at[idx[0], 'Sell_Notes'] = s_note
                    
                    secure_save(df_s, SUMMARY_FILE)
                    
                    st.success(f"Sold. Transaction ID: {s_trx}")
                    st.rerun()
        else: st.info("No positions to sell.")

    # --- TAB 3: UPDATE PRICES ---
    with tab3:
        st.subheader("🛡️ Risk Control Center")
        if st.button("REFRESH MARKET PRICES", type="primary"):
            open_rows = df_s[df_s['Status']=='OPEN']
            if not open_rows.empty:
                p = st.progress(0); n=0
                for i, r in open_rows.iterrows():
                    try:
                        tk = r['Ticker'] if r['Ticker']!='COMP' else '^IXIC'
                        curr = yf.Ticker(tk).history(period='1d')['Close'].iloc[-1]
                        mkt = r['Shares'] * curr
                        unreal = mkt - r['Total_Cost']
                        df_s.at[i, 'Unrealized_PL'] = unreal
                        df_s.at[i, 'Return_Pct'] = (unreal/r['Total_Cost'])*100 if r['Total_Cost'] else 0
                    except: pass
                    n+=1; p.progress(n/len(open_rows))
                secure_save(df_s, SUMMARY_FILE); st.success("✅ Prices Updated!"); st.rerun()
            else: st.warning("No open positions.")

        st.markdown("---")
        st.markdown("### 🛑 Rapid Stop Adjustment")
        open_pos = df_s[df_s['Status'] == 'OPEN'].sort_values('Ticker')
        if not open_pos.empty:
            def get_current_stop_display(tid):
                try:
                    stops = df_d[df_d['Trade_ID'] == tid]['Stop_Loss']
                    val = stops.iloc[-1] if not stops.empty else 0.0
                    return val
                except: return 0.0

            opts_dict = {f"{r['Ticker']} (Current: ${get_current_stop_display(r['Trade_ID']):.2f})": r['Trade_ID'] for _, r in open_pos.iterrows()}
            sel_label = st.selectbox("Select Position to Protect", list(opts_dict.keys()))
            sel_id = opts_dict[sel_label]
            curr_stop_val = get_current_stop_display(sel_id)
            
            c_up1, c_up2, c_up3 = st.columns(3)
            new_stop_price = c_up1.number_input("New Hard Stop Price ($)", value=float(curr_stop_val), min_value=0.0, step=0.01, format="%.2f")
            
            if c_up3.button("UPDATE STOP LOSS"):
                mask = (df_d['Trade_ID'] == sel_id) & (df_d['Action'] == 'BUY')
                if mask.any():
                    last_idx = df_d[mask].last_valid_index()
                    df_d.at[last_idx, 'Stop_Loss'] = new_stop_price
                    secure_save(df_d, DETAILS_FILE)
                    st.success(f"✅ Stop Updated to ${new_stop_price:.2f}")
                    st.rerun()
                else: st.error("Could not find a BUY transaction.")
        else: st.info("No active positions.")

    # --- TAB 4: EDIT TRANSACTION ---
    with tab4:
        st.header("📝 Edit Transaction")
        all_ids = sorted([str(x) for x in df_d['Trade_ID'].unique()], reverse=True)
        if not all_ids:
            st.info("No trades recorded yet.")
        else:
            def fmt_func(x):
                try: return f"{x} | {df_d[df_d['Trade_ID'].astype(str) == x]['Ticker'].iloc[0]}"
                except: return str(x)
            edit_id = st.selectbox("Select Trade ID to Edit", all_ids, format_func=fmt_func)
            if edit_id:
                txs = df_d[df_d['Trade_ID'].astype(str) == edit_id].reset_index().sort_values('Date', ascending=False)
                if not txs.empty:
                    tx_options = [f"{row.get('Trx_ID','')} | {row['Date']} | {row['Action']} {row['Shares']} @ {row['Amount']}" for idx, row in txs.iterrows()]
                    selected_tx_str = st.selectbox("Select Transaction Line", tx_options)
                    if selected_tx_str:
                        sel_idx = tx_options.index(selected_tx_str)
                        row_idx = int(txs.iloc[sel_idx]['index'])
                        current_row = df_d.loc[row_idx]
                        
                        st.markdown("---")
                        cA, cB = st.columns([2, 1])
                        with cA:
                            with st.form("edit_form"):
                                st.subheader(f"Editing: {selected_tx_str}")
                                c1, c2 = st.columns(2)
                                try: dt_obj = pd.to_datetime(current_row['Date'])
                                except: dt_obj = datetime.now()
                                e_date = c1.date_input("Date", dt_obj)
                                e_time = c1.time_input("Time", dt_obj.time(), step=60)
                                
                                # Rule Edit
                                curr_rule = current_row.get('Rule', '')
                                r_idx = ALL_RULES.index(curr_rule) if curr_rule in ALL_RULES else 0
                                e_rule = c2.selectbox("Strategy / Rule", ALL_RULES, index=r_idx)
                                
                                e_trx = st.text_input("Trx ID", value=str(current_row.get('Trx_ID', '')))
                                sl_val = float(current_row['Stop_Loss']) if pd.notna(current_row.get('Stop_Loss')) else 0.0
                                e_stop = c1.number_input("Stop Loss", value=sl_val, step=0.01) 
                                e_note = c2.text_input("Notes", str(current_row.get('Notes', '')))
                                
                                e_shs = c1.number_input("Shares", value=float(current_row['Shares']), step=1.0)
                                e_amt = c2.number_input("Price ($)", value=float(current_row['Amount']), step=0.01)
                                
                                if st.form_submit_button("💾 Save Changes"):
                                    new_ts = datetime.combine(e_date, e_time).strftime("%Y-%m-%d %H:%M")
                                    df_d.at[row_idx, 'Date'] = new_ts
                                    df_d.at[row_idx, 'Rule'] = e_rule
                                    df_d.at[row_idx, 'Stop_Loss'] = e_stop
                                    df_d.at[row_idx, 'Notes'] = e_note
                                    df_d.at[row_idx, 'Shares'] = e_shs
                                    df_d.at[row_idx, 'Amount'] = e_amt
                                    df_d.at[row_idx, 'Value'] = e_shs * e_amt
                                    df_d.at[row_idx, 'Trx_ID'] = e_trx
                                    
                                    secure_save(df_d, DETAILS_FILE)
                                    df_d, df_s = update_campaign_summary(edit_id, df_d, df_s)
                                    secure_save(df_s, SUMMARY_FILE)
                                    st.success("✅ Updated!"); st.rerun()

                        with cB:
                            st.write("### ⚠️ Danger Zone")
                            if st.button("🗑️ DELETE TRANSACTION", type="primary"):
                                df_d = df_d.drop(row_idx)
                                secure_save(df_d, DETAILS_FILE)
                                df_d, df_s = update_campaign_summary(edit_id, df_d, df_s)
                                secure_save(df_s, SUMMARY_FILE)
                                st.warning("Transaction Deleted."); st.rerun()

    # --- TAB 5: DATABASE HEALTH ---
    with tab5:
        st.subheader("Database Maintenance")
        if st.button("FULL REBUILD (Generate Missing Summaries)"):
            if df_d.empty: st.error("Details file is empty.")
            else:
                det_ids = df_d['Trade_ID'].unique()
                sum_ids = df_s['Trade_ID'].unique() if not df_s.empty else []
                missing = [tid for tid in det_ids if tid not in sum_ids]
                new_rows = []
                for tid in missing:
                    trade_txs = df_d[df_d['Trade_ID'] == tid]
                    buys = trade_txs[trade_txs['Action'] == 'BUY'].sort_values('Date')
                    first_tx = buys.iloc[0] if not buys.empty else trade_txs.sort_values('Date').iloc[0]
                    new_rows.append({'Trade_ID': str(tid), 'Ticker': first_tx['Ticker'], 'Status': 'OPEN', 'Open_Date': first_tx['Date'], 'Shares': 0, 'Total_Cost': 0, 'Realized_PL': 0})
                if new_rows:
                    df_s = pd.concat([df_s, pd.DataFrame(new_rows)], ignore_index=True)
                
                all_ids = df_d['Trade_ID'].unique()
                p=st.progress(0)
                for i, tid in enumerate(all_ids):
                    df_d, df_s = update_campaign_summary(tid, df_d, df_s)
                    p.progress((i+1)/len(all_ids))
                
                secure_save(df_d, DETAILS_FILE); secure_save(df_s, SUMMARY_FILE)
                st.success(f"Rebuilt {len(all_ids)} Campaigns."); st.rerun()

    # --- TAB 6: DELETE TRADE ---
    with tab6:
        del_id = st.selectbox("ID to Delete", df_s['Trade_ID'].tolist() if not df_s.empty else [])
        if st.button("DELETE PERMANENTLY"):
            df_s = df_s[df_s['Trade_ID']!=del_id]; df_d = df_d[df_d['Trade_ID']!=del_id]
            secure_save(df_s, SUMMARY_FILE); secure_save(df_d, DETAILS_FILE)
            st.success("Deleted."); st.rerun()

# --- TAB 7: ACTIVE CAMPAIGN SUMMARY (COMPLETE) ---
    with tab7:
        st.subheader("Active Campaign Summary")
        # Ensure we have data
        if not df_s.empty:
             # Filter for OPEN trades
             df_open = df_s[df_s['Status'] == 'OPEN'].copy()
             
             if not df_open.empty:
                 # --- 1. DATA ENRICHMENT ---
                 def get_last_stop(tid):
                     try:
                         rows = df_d[df_d['Trade_ID'] == tid]
                         valid_stops = rows[rows['Stop_Loss'] > 0.01]['Stop_Loss']
                         return valid_stops.iloc[-1] if not valid_stops.empty else 0.0
                     except: return 0.0
                 
                 df_open['Stop Loss'] = df_open['Trade_ID'].apply(get_last_stop)
                 
                 # Financials
                 df_open['Current Value'] = df_open['Total_Cost'] + df_open.get('Unrealized_PL', 0.0).fillna(0.0)
                 
                 # Calculate Current Price based on Value/Shares (handles partials better)
                 df_open['Current Price'] = df_open.apply(lambda x: (x['Current Value']/x['Shares']) if x['Shares'] > 0 else 0, axis=1)
                 
                 # Recalculate Return % explicitly
                 df_open['Return_Pct'] = df_open.apply(lambda x: (x['Unrealized_PL'] / x['Total_Cost'] * 100) if x['Total_Cost'] != 0 else 0.0, axis=1)

                 # --- TREND HUNTER ENGINE (21EMA & 10% RULE) ---
                 def get_trend_status(row):
                     ticker = row['Ticker']
                     avg_cost = row['Avg_Entry']
                     
                     try:
                         # Fetch History (2mo is enough for 21EMA)
                         hist = yf.Ticker(ticker).history(period="2mo")
                         if not hist.empty:
                             hist['21EMA'] = hist['Close'].ewm(span=21, adjust=False).mean()
                             live_ema = hist['21EMA'].iloc[-1]
                             
                             # 1. CHECK "THE GOAL" (Cost < 21EMA)
                             if avg_cost < live_ema:
                                 return f"🏆 WON (Cost < 21e: ${live_ema:.2f})"
                             
                             # 2. CHECK 21e Status
                             curr_px = row['Current Price']
                             if curr_px > live_ema: return f"✅ Above 21e (${live_ema:.2f})"
                             else: return f"⚠️ Below 21e (${live_ema:.2f})"

                     except: return "N/A"
                     return "N/A"

                 df_open['Trend Status'] = df_open.apply(get_trend_status, axis=1)

                 # --- FIXED RISK MATH (CAPITAL AT RISK) ---
                 # Risk = (Avg Entry - Stop) * Shares.
                 # If Stop > Avg Entry, Risk is 0 (Locked Profit).
                 df_open['Risk $'] = (df_open['Avg_Entry'] - df_open['Stop Loss']) * df_open['Shares']
                 df_open['Risk $'] = df_open['Risk $'].apply(lambda x: x if x > 0 else 0.0)
                 
                 # Equity Fetch (Needed for Heat % and Live Exposure)
                 equity = 100000 # Fallback
                 if os.path.exists(JOURNAL_FILE):
                     try: 
                         j_df = pd.read_csv(JOURNAL_FILE)
                         if not j_df.empty:
                            equity = clean_num(j_df['End NLV'].iloc[0]) 
                     except: pass
                 
                 df_open['Risk %'] = (df_open['Risk $'] / equity) * 100
                 df_open['Pos Size %'] = (df_open['Current Value'] / equity) * 100
                 
                 # Max Stop Loss (0.5% Equity Rule) - Theoretical
                 df_open['Max SL (0.5%)'] = df_open.apply(lambda row: row['Avg_Entry'] - ((equity * 0.005) / row['Shares']) if row['Shares'] > 0 else 0, axis=1)
                 
                 # --- 2. THE DASHBOARD METRICS ---
                 total_mkt_val = df_open['Current Value'].sum()
                 total_unreal = df_open['Unrealized_PL'].sum()
                 total_risk = df_open['Risk $'].sum()
                 total_exposure = (total_mkt_val / equity) * 100 if equity > 0 else 0.0
                 
                 # Added Live Exposure Column (m3)
                 m1, m2, m3, m4, m5, m6 = st.columns(6)
                 m1.metric("Open Positions", len(df_open))
                 m2.metric("Total Market Value", f"${total_mkt_val:,.2f}")
                 m3.metric("Live Exposure", f"{total_exposure:.1f}%", f"of ${equity:,.0f}")
                 m4.metric("Total Unrealized P&L", f"${total_unreal:,.2f}", delta_color="normal")
                 m5.metric("Total Open Risk ($)", f"${total_risk:,.2f}")
                 m6.metric("Total Portfolio Heat (%)", f"{df_open['Risk %'].sum():.2f}%")
                 
                 # --- 3. THE DATAFRAME ---
                 # Sort by Highest Return First
                 if 'Return_Pct' in df_open.columns:
                     df_open = df_open.sort_values(by='Return_Pct', ascending=False)

                 cols_target = [
                     'Trade_ID', 'Ticker', 'Trend Status', 'Return_Pct', 'Pos Size %', 
                     'Shares', 'Avg_Entry', 'Current Price', 'Stop Loss', 'Risk_Budget', 
                     'Risk $', 'Risk %', 'Current Value', 'Unrealized_PL', 'Rule'
                 ]
                 cols_final = [c for c in cols_target if c in df_open.columns]
                 
                 st.dataframe(
                     df_open[cols_final].style
                     .format({
                         'Shares':'{:.0f}', 'Total_Cost':'${:,.2f}', 'Unrealized_PL':'${:,.2f}', 
                         'Avg_Entry':'${:,.2f}', 'Current Price':'${:,.2f}', 'Return_Pct':'{:.2f}%', 
                         'Current Value': '${:,.2f}', 'Pos Size %': '{:.1f}%', 'Stop Loss': '${:,.2f}', 
                         'Risk $': '${:,.2f}', 'Risk %': '{:.2f}%', 'Risk_Budget': '${:,.2f}'
                     })
                     .applymap(color_pnl, subset=['Unrealized_PL', 'Return_Pct']),
                     height=(len(df_open) + 1) * 35 + 3,
                     use_container_width=True
                 )

                 # --- 4. RISK MONITOR & GUARDRAILS (UPDATED LOGIC) ---
                 st.markdown("---")
                 st.subheader("🛡️ Risk Monitor")
                 
                 warnings = []
                 all_clear = True
                 
                 for idx, row in df_open.iterrows():
                     ticker = row['Ticker']
                     avg_entry = row['Avg_Entry']
                     shares = row['Shares']
                     curr_price = row['Current Price']
                     curr_stop = row['Stop Loss']
                     curr_risk = row['Risk $']
                     budget = row.get('Risk_Budget', 0.0)
                     ret_pct = row['Return_Pct']

                     # 1. Budget Audit (Risk > Budget)
                     # We allow a tiny $1 buffer for rounding errors
                     if budget > 0 and curr_risk > (budget + 1.0):
                         # Formula: Budget = (Entry - NewStop) * Shares
                         # NewStop = Entry - (Budget / Shares)
                         stop_needed = avg_entry - (budget / shares)
                         warnings.append(f"⚠️ **{ticker}**: Estimated Risk > Budget Risk. **Stop Needed: ${stop_needed:.2f}** (Current Stop: ${curr_stop:.2f})")
                         all_clear = False
                     
                     # 2. Drawdown Violation (>7%)
                     if ret_pct <= -7.0:
                         warnings.append(f"🔴 **{ticker}**: Down {ret_pct:.2f}%. Violates 7-8% Stop Loss Rule. **Immediate Action Required.**")
                         all_clear = False

                     # 3. Profit Protection Rules (Prioritize highest gain)
                     # Thresholds
                     target_15 = avg_entry * 1.15
                     target_10 = avg_entry * 1.10
                     
                     if curr_price >= target_15:
                         # Rule: Move to +5%
                         rec_stop = avg_entry * 1.05
                         if curr_stop < rec_stop:
                             warnings.append(f"🚀 **{ticker}**: Up >15%. Protect Profit. **Move Stop to +5% (${rec_stop:.2f})**. (Current Stop: ${curr_stop:.2f})")
                             all_clear = False
                             
                     elif curr_price >= target_10:
                         # Rule: Move to BE + 0.5% (Slippage Buffer)
                         rec_stop = avg_entry * 1.005 
                         if curr_stop < rec_stop:
                             warnings.append(f"🔔 **{ticker}**: Up >10%. Secure Breakeven. **Move Stop to BE+0.5% (${rec_stop:.2f})**. (Current Stop: ${curr_stop:.2f})")
                             all_clear = False
                
                 if all_clear:
                     st.success("✅ **ALL CLEAR:** System Health Good. All positions are sized within Risk Budgets and Profit Rules.")
                 else:
                     for w in warnings:
                         if "Violates" in w or "Estimated Risk" in w: st.error(w)
                         elif "Up >" in w: st.info(w)
                         else: st.warning(w)

             else: st.info("No open positions.")
        else: st.info("No data available.")

# --- TAB 8: ACTIVE CAMPAIGN DETAILED (STRICT LIFO P&L) ---
    with tab8:
        st.subheader("Active Campaign Detailed (Transactions)")
        if not df_d.empty and not df_s.empty:
            open_ids = df_s[df_s['Status'] == 'OPEN']['Trade_ID'].unique().tolist()
            view_df = df_d[df_d['Trade_ID'].isin(open_ids)].copy()
            
            if not view_df.empty:
                unique_open_tickers = sorted(view_df['Ticker'].unique().tolist())
                tick_filter = st.selectbox("Filter Open Ticker", ["All"] + unique_open_tickers, key='act_det')
                
                # Live Price Override variable
                live_price_override = None 
                
                # --- 1. THE FLIGHT DECK ---
                if tick_filter != "All":
                    summ_row = df_s[(df_s['Ticker'] == tick_filter) & (df_s['Status'] == 'OPEN')]
                    if not summ_row.empty:
                        r = summ_row.iloc[0]
                        try:
                            live_px = yf.Ticker(tick_filter).history(period="1d")['Close'].iloc[-1]
                        except:
                            if r['Shares'] > 0:
                                val = r['Total_Cost'] + r.get('Unrealized_PL', 0)
                                live_px = val / r['Shares']
                            else: live_px = r['Avg_Entry']
                        
                        live_price_override = live_px
                        
                        shares = r['Shares']
                        avg_cost = r['Avg_Entry']
                        realized = r.get('Realized_PL', 0.0)
                        mkt_val = shares * live_px
                        unrealized = mkt_val - r['Total_Cost']
                        return_pct = (unrealized / r['Total_Cost']) * 100 if r['Total_Cost'] else 0.0
                        
                        st.markdown(f"### 🚁 Flight Deck: {tick_filter}")
                        f1, f2, f3, f4, f5, f6 = st.columns(6)
                        f1.metric("Current Price", f"${live_px:,.2f}")
                        f2.metric("Avg Cost", f"${avg_cost:,.2f}")
                        f3.metric("Shares Held", f"{int(shares):,}")
                        f4.metric("Unrealized P&L", f"${unrealized:,.2f}", f"{return_pct:.2f}%")
                        f5.metric("Realized P&L", f"${realized:,.2f}", delta_color="normal")
                        f6.metric("Total Equity", f"${mkt_val:,.2f}")
                        st.markdown("---")
                    
                    view_df = view_df[view_df['Ticker'] == tick_filter]

                # --- 2. DETAILED LIFO ENGINE ---
                if not view_df.empty:
                    start_map = df_s.set_index('Trade_ID')['Open_Date'].to_dict()
                    view_df['Campaign_Start'] = view_df['Trade_ID'].map(start_map)
                    
                    # Fallback prices if needed
                    curr_prices = {}
                    if live_price_override is None:
                        for idx, row in df_s[df_s['Status']=='OPEN'].iterrows():
                            if row['Shares'] > 0: 
                                val = row['Total_Cost'] + row.get('Unrealized_PL', 0)
                                curr_prices[row['Trade_ID']] = val / row['Shares']
                    
                    display_df = view_df.copy()
                    
                    # TRACKERS
                    remaining_map = {}
                    lifo_pl_map = {} # New map to store recalculated P&L
                    
                    for tid in display_df['Trade_ID'].unique():
                        subset = display_df[display_df['Trade_ID'] == tid].copy()
                        # Sort: Date ascending, but Buys (0) before Sells (1)
                        subset['Type_Rank'] = subset['Action'].apply(lambda x: 0 if x == 'BUY' else 1)
                        subset = subset.sort_values(['Date', 'Type_Rank'])
                        
                        inventory = [] 
                        
                        for idx, row in subset.iterrows():
                            if row['Action'] == 'BUY':
                                # Store Price in inventory for P&L calc
                                inventory.append({'idx': idx, 'qty': row['Shares'], 'price': row['Amount']})
                                remaining_map[idx] = row['Shares']
                                
                            elif row['Action'] == 'SELL':
                                to_sell = row['Shares']
                                sell_price = row['Amount']
                                cost_basis_accum = 0.0
                                sold_qty_accum = 0.0
                                
                                # LIFO MATCHING
                                while to_sell > 0 and inventory:
                                    last = inventory.pop() # Pop newest
                                    take = min(to_sell, last['qty'])
                                    
                                    # Math
                                    cost_basis_accum += (take * last['price'])
                                    sold_qty_accum += take
                                    
                                    # Reduce Inventory
                                    last['qty'] -= take
                                    to_sell -= take
                                    remaining_map[last['idx']] = last['qty']
                                    
                                    # Push back remainder
                                    if last['qty'] > 0.00001: inventory.append(last)
                                
                                # CALCULATE TRUE LIFO P&L
                                # P&L = (SellPrice * Qty) - CostBasis
                                revenue = sold_qty_accum * sell_price
                                true_pl = revenue - cost_basis_accum
                                lifo_pl_map[idx] = true_pl

                    # APPLY CALCULATIONS TO DATAFRAME
                    display_df['Remaining_Shares'] = display_df.index.map(remaining_map).fillna(0)
                    
                    # Overwrite Realized_PL with our new LIFO calculation
                    # (Only for SELL rows, BUYs remain 0)
                    display_df['Realized_PL'] = display_df.index.map(lifo_pl_map).fillna(0)
                    
                    # Unrealized P&L (Live)
                    def calc_unrealized(row): 
                         if row['Action'] == 'BUY' and row['Remaining_Shares'] > 0:
                             price = live_price_override if live_price_override is not None else curr_prices.get(row['Trade_ID'], 0)
                             return (price - row['Amount']) * row['Remaining_Shares']
                         return 0.0
                    display_df['Unrealized_PL'] = display_df.apply(calc_unrealized, axis=1)

                    # Return % (Live)
                    def calc_return_pct(row):
                        if row['Action'] == 'BUY' and row['Remaining_Shares'] > 0:
                             price = live_price_override if live_price_override is not None else curr_prices.get(row['Trade_ID'], 0)
                             entry_price = row['Amount']
                             if entry_price > 0: return ((price - entry_price) / entry_price) * 100
                        return 0.0
                    display_df['Return_Pct'] = display_df.apply(calc_return_pct, axis=1)

                    # Visuals
                    display_df['Shares'] = display_df.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
                    display_df['Value'] = display_df.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)
                    
                    final_cols = ['Trade_ID', 'Trx_ID', 'Campaign_Start', 'Date', 'Ticker', 'Action', 'Shares', 'Remaining_Shares', 'Amount', 'Stop_Loss', 'Value', 'Realized_PL', 'Unrealized_PL', 'Return_Pct', 'Rule', 'Notes']
                    show_cols = [c for c in final_cols if c in display_df.columns]
                    
                    st.dataframe(
                        display_df[show_cols].sort_values(['Trade_ID', 'Date']).style
                        .format({
                            'Date': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'None', 
                            'Campaign_Start': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'None'), 
                            'Amount':'${:,.2f}', 'Stop_Loss':'${:,.2f}', 'Value':'${:,.2f}', 
                            'Realized_PL':'${:,.2f}', 'Unrealized_PL':'${:,.2f}', 
                            'Return_Pct':'{:.2f}%', 'Remaining_Shares':'{:.0f}'
                        })
                        .applymap(color_pnl, subset=['Value','Realized_PL','Unrealized_PL', 'Return_Pct'])
                        .applymap(color_neg_value, subset=['Shares']), 
                        height=(len(display_df) + 1) * 35 + 3, 
                        use_container_width=True
                    )
                else: st.info("No matching transactions.")
            else: st.info("No open transactions found.")
        else: st.info("No data available.")

# --- TAB 9: DETAILED TRADE LOG (FINAL: LIFO + TV + TRX_ID) ---
    with tab9:
        st.subheader("🕵️ Campaign Inspector (Post-Mortem)")
        
        # 0. ENSURE JOURNAL IS LOADED
        p_clean = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
        p_legacy = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal.csv')
        path_j = p_clean if os.path.exists(p_clean) else p_legacy
        df_j_hist = pd.DataFrame()
        if os.path.exists(path_j):
            try:
                df_j_hist = pd.read_csv(path_j)
                df_j_hist['Day'] = pd.to_datetime(df_j_hist['Day'], errors='coerce')
                df_j_hist = df_j_hist.sort_values('Day', ascending=False)
                
                def clean_nlv_val(x):
                    try: return float(str(x).replace('$', '').replace(',', '').strip())
                    except: return 0.0
                if 'End NLV' in df_j_hist.columns:
                    df_j_hist['End NLV'] = df_j_hist['End NLV'].apply(clean_nlv_val)
            except: pass

        # 1. TWO-STAGE FILTER
        all_tickers = sorted(df_d['Ticker'].dropna().unique().tolist())
        
        c_filt1, c_filt2 = st.columns(2)
        sel_tick = c_filt1.selectbox("1. Select Ticker", ["All"] + all_tickers)
        
        view_df = pd.DataFrame()
        sel_id = None
        
        if sel_tick != "All":
            subset_d = df_d[df_d['Ticker'] == sel_tick]
            subset_s = df_s[df_s['Ticker'] == sel_tick]
            
            trade_ids = sorted(subset_d['Trade_ID'].unique().tolist(), reverse=True)
            sel_id = c_filt2.selectbox("2. Select Campaign ID", trade_ids)
            
            if sel_id:
                # Filter specifically for this ID
                camp_txs = subset_d[subset_d['Trade_ID'] == sel_id].sort_values('Date')
                
                # --- A. RUN LIFO ENGINE FIRST (TO GET TRUE P&L) ---
                calc_df = camp_txs.copy().reset_index()
                buy_attribution = {} 
                inventory = [] 
                
                for idx, row in calc_df.iterrows():
                    if row['Action'] == 'BUY':
                        inventory.append({'idx': idx, 'price': row['Amount'], 'qty': row['Shares']})
                        buy_attribution[idx] = {'pl': 0.0, 'sold_cost': 0.0, 'sold_val': 0.0}
                    elif row['Action'] == 'SELL':
                        to_sell = row['Shares']
                        sell_price = row['Amount']
                        while to_sell > 0 and inventory:
                            last = inventory.pop()
                            take = min(to_sell, last['qty'])
                            seg_cost = take * last['price']
                            seg_rev = take * sell_price
                            seg_pl = seg_rev - seg_cost
                            buy_attribution[last['idx']]['pl'] += seg_pl
                            buy_attribution[last['idx']]['sold_cost'] += seg_cost
                            buy_attribution[last['idx']]['sold_val'] += seg_rev
                            last['qty'] -= take
                            to_sell -= take
                            if last['qty'] > 0.0001: inventory.append(last)

                def get_lifo_pl(idx, action, original_pl):
                    if action == 'SELL': return original_pl 
                    if idx in buy_attribution: return buy_attribution[idx]['pl']
                    return 0.0

                def get_lifo_ret(idx, action):
                    if action == 'BUY' and idx in buy_attribution:
                        data = buy_attribution[idx]
                        if data['sold_cost'] > 0:
                            return ((data['sold_val'] - data['sold_cost']) / data['sold_cost']) * 100
                    return 0.0

                calc_df['Lot P&L'] = calc_df.apply(lambda x: get_lifo_pl(x.name, x['Action'], x['Realized_PL']), axis=1)
                calc_df['Return %'] = calc_df.apply(lambda x: get_lifo_ret(x.name, x['Action']), axis=1)
                
                # --- B. CALCULATE METRICS ---
                realized_pl = calc_df[calc_df['Action'] == 'BUY']['Lot P&L'].sum()
                
                camp_sum = subset_s[subset_s['Trade_ID'] == sel_id].iloc[0] if not subset_s.empty else pd.Series()
                start_date = pd.to_datetime(calc_df['Date'].iloc[0])
                last_date = pd.to_datetime(calc_df['Date'].iloc[-1])
                
                is_closed = False
                if not camp_sum.empty and camp_sum['Status'] == 'CLOSED':
                    is_closed = True
                    if pd.notnull(camp_sum['Closed_Date']):
                        last_date = pd.to_datetime(camp_sum['Closed_Date'])
                else: last_date = datetime.now()
                
                days_held = (last_date - start_date).days
                if days_held < 1: days_held = 1

                # Risk Budget
                risk_budget = camp_sum.get('Risk_Budget', 0.0)
                risk_source = "Locked"
                if risk_budget <= 0:
                    risk_source = "Est. (0.5% NLV)"
                    if not df_j_hist.empty:
                        prior_days = df_j_hist[df_j_hist['Day'] < start_date]
                        if not prior_days.empty:
                            risk_budget = prior_days.iloc[0]['End NLV'] * 0.005
                        else: risk_budget = 500.0
                    else: risk_budget = 500.0
                
                r_str = "N/A"; r_color = "off"
                if risk_budget > 0:
                    r_multiple = realized_pl / risk_budget
                    r_str = f"{r_multiple:+.2f}R"
                    r_color = "normal" if r_multiple > 0 else "inverse"

                # Efficiency
                mfe_str = "N/A"
                try:
                    chart_start = start_date - timedelta(days=5)
                    chart_end = last_date + timedelta(days=5)
                    chart_data = yf.Ticker(sel_tick).history(start=chart_start, end=chart_end)
                    
                    if not chart_data.empty:
                        hold_mask = (chart_data.index >= start_date.tz_localize(chart_data.index.tz)) & (chart_data.index <= last_date.tz_localize(chart_data.index.tz))
                        if any(hold_mask):
                             period_high = chart_data.loc[hold_mask]['High'].max()
                             sells = calc_df[calc_df['Action'] == 'SELL']
                             if not sells.empty:
                                avg_exit = (sells['Amount'] * sells['Shares']).sum() / sells['Shares'].sum()
                                efficiency = (avg_exit / period_high) * 100
                                mfe_str = f"{efficiency:.1f}% (High: ${period_high:.2f})"
                             elif is_closed: mfe_str = "0% (Stopped Out?)"
                except: pass

                # --- C. DISPLAY FLIGHT DECK ---
                st.markdown(f"### 🚁 Flight Deck: {sel_tick} ({sel_id})")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Realized P&L", f"${realized_pl:+,.2f}", f"{days_held} Days Held")
                m2.metric("R-Multiple", r_str, f"Risk Base: ${risk_budget:,.0f} ({risk_source})", delta_color=r_color)
                
                buys = calc_df[calc_df['Action'] == 'BUY']
                avg_in = (buys['Amount'] * buys['Shares']).sum() / buys['Shares'].sum() if not buys.empty else 0
                m3.metric("Avg Entry Price", f"${avg_in:.2f}")
                m4.metric("Exit Efficiency", mfe_str, "vs Period High")
                
                # --- D. TRADINGVIEW EMBED (THE BATTLEFIELD) ---
                st.markdown("### 🗺️ The Battlefield (TradingView)")
                tv_widget_code = f"""
                <div class="tradingview-widget-container">
                  <div id="tradingview_chart"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                  <script type="text/javascript">
                  new TradingView.widget(
                  {{
                    "width": "100%",
                    "height": 500,
                    "symbol": "{sel_tick}",
                    "interval": "D",
                    "timezone": "America/New_York",
                    "theme": "light",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#f1f3f6",
                    "enable_publishing": false,
                    "hide_side_toolbar": false,
                    "allow_symbol_change": true,
                    "container_id": "tradingview_chart"
                  }}
                  );
                  </script>
                </div>
                """
                st.components.v1.html(tv_widget_code, height=500)
                
                # DEEP LINK
                tv_link = f"https://www.tradingview.com/chart/?symbol={sel_tick}"
                st.link_button(f"🚀 Analyze {sel_tick} on TradingView (Premium)", tv_link)
                st.markdown("---")

                # --- E. NARRATIVE ---
                n1, n2 = st.columns(2)
                with n1:
                    st.info(f"**📝 Buy Rationale:**\n{camp_sum.get('Buy_Notes', 'No notes.')}")
                    st.caption(f"Strategy: {camp_sum.get('Rule', 'N/A')}")
                with n2:
                    sell_note = camp_sum.get('Sell_Notes', '')
                    sell_rule = camp_sum.get('Sell_Rule', '')
                    if not sell_note and not sell_rule:
                        st.warning("**No Exit Plan/Notes Logged**")
                    else:
                        st.error(f"**👋 Exit Context:**\n{sell_note}")
                        st.caption(f"Exit Rule: {sell_rule}")

                # --- F. RENDER TABLE ---
                st.markdown("#### 📜 Transaction History (LIFO Attribution)")
                display_df = calc_df.copy()
                
                # Visuals
                display_df['Shares'] = display_df.apply(lambda x: -x['Shares'] if x['Action'] == 'SELL' else x['Shares'], axis=1)
                display_df['Value'] = display_df.apply(lambda x: -x['Value'] if x['Action'] == 'SELL' else x['Value'], axis=1)
                
                # ADDED 'Trx_ID' back to the list
                cols = ['Trade_ID', 'Trx_ID', 'Date', 'Ticker', 'Action', 'Shares', 'Amount', 'Value', 'Lot P&L', 'Return %', 'Rule', 'Notes']
                show_cols = [c for c in cols if c in display_df.columns]
                
                st.dataframe(
                    display_df[show_cols].sort_values(['Trade_ID', 'Date']).style
                    .format({
                        'Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'None', 
                        'Shares':'{:.0f}', 'Amount':'${:,.2f}', 'Value':'${:,.2f}', 
                        'Lot P&L':'${:,.2f}', 'Return %':'{:.2f}%'
                    })
                    .applymap(color_pnl, subset=['Lot P&L', 'Return %'])
                    .applymap(color_neg_value, subset=['Shares']),
                    use_container_width=True
                )
        else:
            view_df = df_d.copy()
            view_df['Lot P&L'] = view_df['Realized_PL']
            view_df['Return %'] = 0.0
            st.markdown("### 🗄️ Master Transaction Log")
            st.dataframe(view_df.sort_values(['Date'], ascending=False), use_container_width=True)

# --- TAB CY: CURRENT YEAR CAMPAIGNS (2026 + ROLLOVERS) ---
    with tab_cy:
        st.subheader("CY 2026 Campaigns (Risk & Performance)")
        st.caption("Showing 2026 trades + Rollovers. Auditing both Risk Discipline and Financial Performance.")

        if not df_s.empty:
            # --- 1. FILTER LOGIC ---
            df_s['Open_DT'] = pd.to_datetime(df_s['Open_Date'], errors='coerce')
            df_s['Close_DT'] = pd.to_datetime(df_s['Closed_Date'], errors='coerce')
            cutoff_date = pd.Timestamp("2026-01-01")
            
            cy_mask = (
                (df_s['Open_DT'] >= cutoff_date) | 
                (df_s['Status'] == 'OPEN') | 
                (df_s['Close_DT'] >= cutoff_date)
            )
            df_cy = df_s[cy_mask].copy()
            
            if not df_cy.empty:
                # --- 2. CALCULATE METRICS (Combined Engine) ---
                df_cy = df_cy.reset_index().rename(columns={'index': 'Seq_ID'})
                
                # Ensure Risk_Budget exists
                if 'Risk_Budget' not in df_cy.columns: df_cy['Risk_Budget'] = 0.0
                df_cy['Risk_Budget'] = df_cy['Risk_Budget'].fillna(0.0).astype(float)
                
                def calc_row_metrics(row):
                    # P&L Logic
                    pl = row['Realized_PL'] if row['Status'] == 'CLOSED' else row.get('Unrealized_PL', 0.0)
                    
                    # Risk Logic
                    budget = row['Risk_Budget']
                    if budget > 0:
                        r_mult = pl / budget
                    else:
                        r_mult = 0.0
                    
                    # Compliance Logic (Losses Only)
                    compliance = "N/A"
                    if pl >= 0:
                        compliance = "✅ WIN"
                    else:
                        if budget > 0:
                            loss_ratio = abs(pl) / budget
                            if loss_ratio <= 1.1: compliance = "✅ OK"      
                            elif loss_ratio <= 1.5: compliance = "⚠️ SLIP"  
                            else: compliance = "🛑 BREACH"                  
                        else:
                            compliance = "⚪ NO BUDGET"
                    
                    return pd.Series([pl, r_mult, compliance])

                df_cy[['Active_PL', 'R_Multiple', 'Compliance']] = df_cy.apply(calc_row_metrics, axis=1)

                # --- 3. FILTERS ---
                c_f1, c_f2 = st.columns(2)
                unique_tickers_cy = sorted(df_cy['Ticker'].dropna().astype(str).unique().tolist())
                tick_filter_cy = c_f1.selectbox("Filter Ticker (CY)", ["All"] + unique_tickers_cy, key="cy_tick")
                comp_filter = c_f2.multiselect("Filter Compliance", ["✅ WIN", "✅ OK", "⚠️ SLIP", "🛑 BREACH"], key="cy_comp")
                
                view_cy = df_cy.copy()
                if tick_filter_cy != "All": view_cy = view_cy[view_cy['Ticker'] == tick_filter_cy]
                if comp_filter: view_cy = view_cy[view_cy['Compliance'].isin(comp_filter)]
                
                if not view_cy.empty:
                    # --- 4. FLIGHT DECK (RESTORED & EXPANDED) ---
                    closed_cy = view_cy[view_cy['Status'] == 'CLOSED']
                    
                    # Defaults
                    net_pl = view_cy['Active_PL'].sum() # Active P&L (Open + Closed)
                    win_rate = 0.0; expectancy = 0.0
                    gross_profit = 0.0; gross_loss = 0.0
                    avg_win = 0.0; avg_loss = 0.0
                    avg_r_loss = 0.0; discipline_score = 0.0
                    wl_ratio = 0.0; num_wins = 0; num_losses = 0

                    if not closed_cy.empty:
                        # Financials
                        winners = closed_cy[closed_cy['Active_PL'] > 0]
                        losers = closed_cy[closed_cy['Active_PL'] <= 0]
                        
                        gross_profit = winners['Active_PL'].sum()
                        gross_loss = abs(losers['Active_PL'].sum())
                        
                        num_wins = len(winners)
                        num_losses = len(losers)
                        total_closed = len(closed_cy)
                        
                        win_rate = (num_wins / total_closed) * 100 if total_closed > 0 else 0.0
                        avg_win = gross_profit / num_wins if num_wins > 0 else 0.0
                        avg_loss = gross_loss / num_losses if num_losses > 0 else 0.0
                        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
                        
                        # Expectancy
                        win_pct_dec = win_rate / 100
                        loss_pct_dec = 1 - win_pct_dec
                        expectancy = (win_pct_dec * avg_win) - (loss_pct_dec * avg_loss)
                        
                        # Risk Auditing
                        avg_r_loss = losers['R_Multiple'].mean() if not losers.empty else 0.0
                        if not losers.empty:
                            compliant_losers = losers[losers['Compliance'] == '✅ OK']
                            discipline_score = (len(compliant_losers) / len(losers)) * 100
                        else: discipline_score = 100.0

                    # RENDER SCOREBOARD (MATCHING TAB 10 STYLE)
                    st.markdown("### 🚁 Flight Deck: Performance & Risk")
                    
                    # Row 1: The Bottom Line (Added Deltas)
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Net P&L (CY)", f"${net_pl:,.2f}", f"{len(view_cy)} Total Campaigns")
                    m2.metric("Win Rate", f"{win_rate:.1f}%", f"{num_wins}W - {num_losses}L")
                    m3.metric("Expectancy", f"${expectancy:,.2f}", "Avg value per trade")
                    
                    # Risk Deltas (Custom for Risk)
                    disc_delta = "Perfect" if discipline_score == 100 else ("Needs Work" if discipline_score < 80 else "Solid")
                    disc_col = "normal" if discipline_score >= 90 else "inverse"
                    m4.metric("Risk Compliance", f"{discipline_score:.0f}%", disc_delta, delta_color=disc_col)
                    
                    loss_col = "normal" if avg_r_loss > -1.2 else "inverse"
                    m5.metric("Avg R-Loss", f"{avg_r_loss:.2f}R", "Target: > -1.0R", delta_color=loss_col)
                    
                    st.markdown("---")
                    
                    # Row 2: Dollar Stats (Added Deltas)
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Gross Profit", f"${gross_profit:,.2f}", delta_color="normal")
                    k2.metric("Gross Loss", f"-${gross_loss:,.2f}", delta_color="inverse")
                    k3.metric("Avg Win", f"${avg_win:,.2f}", delta_color="normal")
                    k4.metric("Avg Loss", f"-${avg_loss:,.2f}", f"W/L Ratio: {wl_ratio:.2f}")

                    st.markdown("---")

                    # --- 5. DATA TABLE (REORDERED & COLORED) ---
                    def calc_days_open(row):
                        try:
                            start = row['Open_DT']
                            end = row['Close_DT'] if row['Status'] == 'CLOSED' and pd.notna(row['Close_DT']) else datetime.now()
                            return (end - start).days
                        except: return 0
                    view_cy['Days_Open'] = view_cy.apply(calc_days_open, axis=1)

                    # NEW COLUMN ORDER
                    target_cols = [
                        'Seq_ID', 'Trade_ID', 'Ticker', 'Status', 
                        'Open_Date', 'Closed_Date', 'Days_Open', 
                        'Total_Cost', 'Avg_Entry', 'Avg_Exit', 
                        'Risk_Budget', 'Active_PL', 'R_Multiple', 
                        'Compliance', 'Rule', 'Buy_Notes', 'Sell_Notes'
                    ]
                    # Ensure cols exist
                    for c in ['Avg_Exit', 'Buy_Notes', 'Sell_Notes']:
                        if c not in view_cy.columns: view_cy[c] = ""
                            
                    valid_cols = [c for c in target_cols if c in view_cy.columns]
                    
                    view_cy = view_cy.sort_values('Open_DT', ascending=False)
                    
                    # --- STYLING FUNCTIONS ---
                    def style_status(val): 
                        if val == 'CLOSED': return 'color: #ff4b4b; font-weight: bold' # Red
                        return 'color: #2ca02c; font-weight: bold' # Green
                        
                    def style_pl(val):
                        if val > 0: return 'color: #2ca02c'
                        if val < 0: return 'color: #ff4b4b'
                        return ''
                        
                    def style_compliance(val):
                        if 'BREACH' in str(val): return 'color: white; background-color: #ff4b4b; font-weight: bold' 
                        if 'SLIP' in str(val): return 'color: #ff4b4b; font-weight: bold' 
                        if 'WIN' in str(val): return 'color: #2ca02c; font-weight: bold'
                        return ''
                    
                    def style_r(val):
                        if val > 1.0: return 'color: #2ca02c; font-weight: bold'
                        if val < -1.2: return 'color: #ff4b4b; font-weight: bold'
                        return ''

                    st.dataframe(
                        view_cy[valid_cols].style.format({
                            'Open_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '',
                            'Closed_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '',
                            'Total_Cost':'${:,.2f}', 'Avg_Entry':'${:,.2f}', 'Avg_Exit':'${:,.2f}',
                            'Risk_Budget':'${:,.0f}', 'Active_PL':'${:+,.2f}', 'R_Multiple':'{:+.2f}R'
                        })
                        .applymap(style_status, subset=['Status'])
                        .applymap(style_pl, subset=['Active_PL'])
                        .applymap(style_compliance, subset=['Compliance'])
                        .applymap(style_r, subset=['R_Multiple']),
                        hide_index=True,
                        use_container_width=True
                    )
                else: st.info("No trades match your filters.")
            else: st.info("No trades found for 2026 or Rollovers.")
        else: st.warning("Summary Database is empty.")

# --- TAB 10: ALL CAMPAIGNS (PRO SCOREBOARD) ---
    with tab10:
        st.subheader("All Campaigns (Summary)")
        
        # 1. Prepare Data
        df_s_view = df_s.reset_index().rename(columns={'index': 'Seq_ID'})
        
        def get_result(row):
            if row['Status'] == 'OPEN': return "OPEN"
            pct = row['Return_Pct']
            return "BE" if -0.5 <= pct <= 0.5 else ("WIN" if pct > 0.5 else "LOSS")
        df_s_view['Result'] = df_s_view.apply(get_result, axis=1)
        
        # 2. Filters
        unique_tickers_sum = sorted(df_s['Ticker'].dropna().astype(str).unique().tolist())
        tick_filter_all = st.selectbox("Filter Campaign Ticker", ["All"] + unique_tickers_sum)
        
        unique_rules = sorted([str(x) for x in df_s['Rule'].unique() if pd.notnull(x)])
        rule_filter = st.multiselect("Filter by Buy Rule", unique_rules)
        res_filter = st.multiselect("Filter by Result", ["WIN", "LOSS", "BE", "OPEN"])
        
        view_all = df_s_view.copy()
        
        # Apply Filters
        if tick_filter_all != "All": view_all = view_all[view_all['Ticker'] == tick_filter_all]
        if rule_filter: view_all = view_all[view_all['Rule'].isin(rule_filter)]
        if res_filter: view_all = view_all[view_all['Result'].isin(res_filter)]
        
        if not view_all.empty:
            # --- 3. THE SCOREBOARD (METRICS ENGINE) ---
            closed_trades = view_all[view_all['Status'] == 'CLOSED']
            
            if not closed_trades.empty:
                # Basic Counts
                total_trades = len(closed_trades)
                wins = closed_trades[closed_trades['Result'] == 'WIN']
                losses = closed_trades[closed_trades['Result'] == 'LOSS']
                
                num_wins = len(wins)
                num_losses = len(losses)
                win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0.0
                
                # Dollar Stats
                gross_profit = wins['Realized_PL'].sum()
                gross_loss = abs(losses['Realized_PL'].sum())
                net_pl = gross_profit - gross_loss
                
                # Averages
                avg_win = gross_profit / num_wins if num_wins > 0 else 0.0
                avg_loss = gross_loss / num_losses if num_losses > 0 else 0.0
                
                # Key Ratios
                pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
                
                # Expectancy (Average R per trade)
                # Formula: (Win% * AvgWin) - (Loss% * AvgLoss)
                win_pct_dec = win_rate / 100
                loss_pct_dec = 1 - win_pct_dec
                expectancy = (win_pct_dec * avg_win) - (loss_pct_dec * avg_loss)
                
                # --- RENDER METRICS ---
                st.markdown("### 🏆 Performance Matrix (Closed Trades)")
                
                # Row 1: The Bottom Line
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Net Profit", f"${net_pl:,.2f}", f"{len(view_all)} Total Campaigns")
                m2.metric("Profit Factor", f"{pf:.2f}", delta="Excellent" if pf > 2.0 else "Needs Work" if pf < 1.0 else "Good")
                m3.metric("Win Rate", f"{win_rate:.1f}%", f"{num_wins}W - {num_losses}L")
                m4.metric("Expectancy", f"${expectancy:,.2f}", "Avg value per trade")
                
                st.markdown("---")
                
                # Row 2: The Edge
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Gross Profit", f"${gross_profit:,.2f}", delta_color="normal")
                k2.metric("Gross Loss", f"-${gross_loss:,.2f}", delta_color="inverse")
                k3.metric("Avg Win", f"${avg_win:,.2f}", delta_color="normal")
                k4.metric("Avg Loss", f"-${avg_loss:,.2f}", f"W/L Ratio: {wl_ratio:.2f}")
                
                st.markdown("---")
            else:
                st.info("No closed trades in this view to calculate metrics.")

            # 4. Duration Calculation
            def calc_days_open(row):
                try:
                    start = pd.to_datetime(row['Open_Date'])
                    end = pd.to_datetime(row['Closed_Date']) if row['Status'] == 'CLOSED' and pd.notna(row['Closed_Date']) else datetime.now()
                    return (end - start).days
                except: return 0
            view_all['Days_Open'] = view_all.apply(calc_days_open, axis=1)
            
            # 5. Table Display
            all_cols = [
                'Seq_ID', 'Trade_ID', 'Ticker', 'Status', 'Result', 
                'Open_Date', 'Closed_Date', 'Days_Open', 
                'Shares', 'Avg_Entry', 'Avg_Exit', 'Total_Cost', 
                'Realized_PL', 'Return_Pct', 
                'Rule', 'Buy_Notes', 'Sell_Rule', 'Sell_Notes'
            ]
            valid_all = [c for c in all_cols if c in df_s_view.columns]
            
            # Sort by Sequence (preserve original order)
            view_all = view_all.sort_values('Seq_ID', ascending=False)
            
            def highlight_status(val): return 'color: red' if val == 'CLOSED' else 'color: green'
            
            st.dataframe(
                view_all[valid_all].style.format({
                    'Open_Date': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d') if pd.notnull(x) else 'None'), 
                    'Closed_Date': lambda x: x if isinstance(x, str) else (x.strftime('%Y-%m-%d') if pd.notnull(x) else 'None'), 
                    'Shares':'{:.0f}', 'Avg_Entry':'${:,.2f}', 'Avg_Exit':'${:,.2f}', 
                    'Total_Cost':'${:,.2f}', 'Realized_PL':'${:,.2f}', 'Return_Pct':'{:.2f}%'
                })
                .applymap(highlight_status, subset=['Status'])
                .applymap(color_pnl, subset=['Realized_PL'])
                .applymap(color_result, subset=['Result']), 
                hide_index=True,
                use_container_width=True
            )
        else: st.info("No campaigns match your filters.")

# --- TAB 11: PERFORMANCE AUDIT (WITH PERIOD SELECTOR & FIXED MATH) ---
    with tab11:
        st.subheader("🏆 Performance Audit: The Best & The Worst")
        st.markdown("Analysis of outlier trades to determine 'R' efficiency and P&L concentration.")
        
        # Load necessary data
        if not df_s.empty:
            
            # --- 1. PERIOD SELECTOR ---
            c_scope1, c_scope2 = st.columns(2)
            scope_mode = c_scope1.selectbox("Analysis Period", ["All Time", "Current Year (YTD)", "Previous Year", "Custom Range"])
            
            # Filter Logic
            audit_source = df_s[df_s['Status'] == 'CLOSED'].copy()
            audit_source['Closed_Date'] = pd.to_datetime(audit_source['Closed_Date'], errors='coerce')
            
            start_d, end_d = None, None
            now = datetime.now()
            
            if scope_mode == "Current Year (YTD)":
                start_d = datetime(now.year, 1, 1)
                end_d = now
            elif scope_mode == "Previous Year":
                start_d = datetime(now.year - 1, 1, 1)
                end_d = datetime(now.year - 1, 12, 31)
            elif scope_mode == "Custom Range":
                d_range = c_scope2.date_input("Select Range", [now - timedelta(days=90), now])
                if len(d_range) == 2:
                    start_d, end_d = datetime.combine(d_range[0], datetime.min.time()), datetime.combine(d_range[1], datetime.max.time())

            # Apply Filter
            if start_d and end_d:
                # Filter by Closed Date
                audit_df = audit_source[
                    (audit_source['Closed_Date'] >= start_d) & 
                    (audit_source['Closed_Date'] <= end_d)
                ].copy()
                st.caption(f"Showing trades closed between {start_d.strftime('%Y-%m-%d')} and {end_d.strftime('%Y-%m-%d')}")
            else:
                audit_df = audit_source.copy()
                st.caption("Showing ALL closed trades.")
            
            st.markdown("---")

            if not audit_df.empty:
                if st.button("🚀 RUN AUDIT", type="primary"):
                    
                    # 2. PREPARE HISTORY FOR NLV LOOKUP
                    p_clean = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal_Clean.csv')
                    p_legacy = os.path.join(DATA_ROOT, portfolio, 'Trading_Journal.csv')
                    path_j = p_clean if os.path.exists(p_clean) else p_legacy
                    
                    df_j_hist = pd.DataFrame()
                    if os.path.exists(path_j):
                        try:
                            df_j_hist = pd.read_csv(path_j)
                            df_j_hist['Day'] = pd.to_datetime(df_j_hist['Day'], errors='coerce')
                            df_j_hist = df_j_hist.sort_values('Day', ascending=True)
                            
                            def clean_nlv_audit(x):
                                try: return float(str(x).replace('$', '').replace(',', '').strip())
                                except: return 0.0
                            if 'End NLV' in df_j_hist.columns:
                                df_j_hist['End NLV'] = df_j_hist['End NLV'].apply(clean_nlv_audit)
                        except: pass

                    # 3. CALCULATION ENGINE
                    results = []
                    progress_bar = st.progress(0)
                    total_rows = len(audit_df)
                    
                    for i, (idx, row) in enumerate(audit_df.iterrows()):
                        progress_bar.progress((i + 1) / total_rows)
                        
                        # A. Risk Budget & R-Multiple
                        budget = row.get('Risk_Budget', 0.0)
                        
                        if budget <= 0:
                            open_date = pd.to_datetime(row['Open_Date'])
                            if not df_j_hist.empty:
                                prior = df_j_hist[df_j_hist['Day'] < open_date]
                                if not prior.empty:
                                    budget = prior.iloc[-1]['End NLV'] * 0.005
                                else: budget = 500.0
                            else: budget = 500.0
                        
                        realized = row['Realized_PL']
                        r_mult = realized / budget if budget > 0 else 0.0
                        
                        # B. Exit Efficiency
                        eff_val = 0.0
                        try:
                            o_date = pd.to_datetime(row['Open_Date']).tz_localize(None)
                            c_date = row['Closed_Date'].tz_localize(None) if pd.notnull(row['Closed_Date']) else datetime.now()
                            
                            h_data = yf.Ticker(row['Ticker']).history(start=o_date, end=c_date + timedelta(days=1))
                            if not h_data.empty:
                                period_high = h_data['High'].max()
                                if row['Shares'] > 0:
                                    calc_exit = (row['Realized_PL'] / row['Shares']) + row['Avg_Entry']
                                    if period_high > 0:
                                        eff_val = (calc_exit / period_high) * 100
                        except: pass
                        
                        results.append({
                            'Trade_ID': row['Trade_ID'],
                            'Ticker': row['Ticker'],
                            'Open_Date': row['Open_Date'],
                            'Closed_Date': row['Closed_Date'], # <--- ADDED CLOSED DATE
                            'Net P&L': realized,
                            'Return %': row.get('Return_Pct', 0.0),
                            'Risk Budget': budget,
                            'R-Multiple': r_mult,
                            'Exit Eff %': eff_val
                        })
                    
                    res_df = pd.DataFrame(results)
                    progress_bar.empty()
                    
                    # 4. SORTING
                    top_15 = res_df.sort_values('Net P&L', ascending=False).head(15)
                    bot_15 = res_df.sort_values('Net P&L', ascending=True).head(15)
                    
                    # 5. AGGREGATE STATS (CORRECTED MATH)
                    # Calculate Total Gross Profit (Sum of all positives) and Total Gross Loss (Sum of all negatives)
                    gross_profit = res_df[res_df['Net P&L'] > 0]['Net P&L'].sum()
                    gross_loss = res_df[res_df['Net P&L'] < 0]['Net P&L'].sum() # This is a negative number
                    net_pl = gross_profit + gross_loss
                    
                    top_sum = top_15['Net P&L'].sum()
                    bot_sum = bot_15['Net P&L'].sum()
                    
                    # Ratios
                    pct_top_of_gross = (top_sum / gross_profit * 100) if gross_profit != 0 else 0
                    pct_bot_of_loss = (bot_sum / gross_loss * 100) if gross_loss != 0 else 0 # e.g. -15k / -20k = 75%
                    
                    # --- DISPLAY METRICS ---
                    st.markdown("### 📊 Concentration Analysis (Pareto)")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Net P&L (Period)", f"${net_pl:,.2f}")
                    c2.metric("Total Gross Profit", f"${gross_profit:,.2f}")
                    c3.metric("Total Gross Loss", f"${gross_loss:,.2f}")
                    c4.metric("Profit Factor", f"{abs(gross_profit/gross_loss):.2f}" if gross_loss != 0 else "Inf")
                    
                    st.markdown("#### Outlier Impact")
                    k1, k2 = st.columns(2)
                    k1.metric("Top 15 Winners Sum", f"${top_sum:,.2f}", f"{pct_top_of_gross:.1f}% of Gross Profit")
                    k2.metric("Bottom 15 Losers Sum", f"${bot_sum:,.2f}", f"{pct_bot_of_loss:.1f}% of Gross Loss", delta_color="inverse")
                    
                    st.markdown("---")
                    
                    # --- TOP 15 TABLE ---
                    st.subheader("🟢 Top 15 Best Trades")
                    st.dataframe(
                        top_15.style
                        .format({
                            'Net P&L': '${:,.2f}', 'Return %': '{:.2f}%', 
                            'Risk Budget': '${:,.0f}', 'R-Multiple': '{:+.2f}R',
                            'Exit Eff %': '{:.1f}%',
                            'Open_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '',
                            'Closed_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else ''
                        })
                        .applymap(lambda x: 'color: #4CAF50' if x > 0 else 'color: #FF5252', subset=['Net P&L', 'R-Multiple', 'Return %']),
                        use_container_width=True,
                        height=550
                    )
                    
                    st.markdown("---")
                    
                    # --- BOTTOM 15 TABLE ---
                    st.subheader("🔴 Top 15 Worst Trades")
                    st.dataframe(
                        bot_15.style
                        .format({
                            'Net P&L': '${:,.2f}', 'Return %': '{:.2f}%', 
                            'Risk Budget': '${:,.0f}', 'R-Multiple': '{:+.2f}R',
                            'Exit Eff %': '{:.1f}%',
                            'Open_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else '',
                            'Closed_Date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if pd.notnull(x) else ''
                        })
                        .applymap(lambda x: 'color: #4CAF50' if x > 0 else 'color: #FF5252', subset=['Net P&L', 'R-Multiple', 'Return %']),
                        use_container_width=True,
                        height=550
                    )
                    
            else:
                st.info("No closed trades found for this period.")
        else:
            st.warning("Summary file empty.")