import numpy as np

def performance_report(trades):
    if not trades: return {}
    R_list = [t['R'] for t in trades]
    wins = [r for r in R_list if r > 0]; losses = [r for r in R_list if r <= 0]
    win_rate = (len(wins)/len(R_list))*100.0 if R_list else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    expectancy = float(np.mean(R_list)) if R_list else 0.0
    eq = np.cumsum(R_list); peak = np.maximum.accumulate(eq); dd = eq - peak; mdd = float(dd.min()) if len(dd)>0 else 0.0
    profit_factor = (sum([r for r in R_list if r>0]) / abs(sum([r for r in R_list if r<=0])) ) if sum([r for r in R_list if r<=0])!=0 else float('inf')
    return {
        "trades": len(R_list),
        "win_rate_%": round(win_rate,2),
        "avg_win_R": round(avg_win,3),
        "avg_loss_R": round(avg_loss,3),
        "expectancy_R": round(expectancy,3),
        "MDD_R": round(mdd,3),
        "profit_factor": round(profit_factor,3),
        "total_R": round(float(eq[-1]),3),
    }
