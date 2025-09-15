from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title="Bankruptcy Prediction")

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"]
)

# Cargar modelo entrenado
model = load(pathlib.Path("model/model-v1.joblib"))

class InputData(BaseModel):
    ROA_C: float = 0.370594257300249
    ROA_A: float = 0.424389446140427
    ROA_B: float = 0.40574977247176
    Operating_Gross_Margin: float = 0.601457213277793
    Realized_Sales_Gross_Margin: float = 0.601457213277793
    Operating_Profit_Rate: float = 0.998969203197885
    Pre_tax_net_Interest_Rate: float = 0.796887145860514
    After_tax_net_Interest_Rate: float = 0.808809360876843
    Non_industry_income_and_expenditure_revenue: float = 0.302646433889668
    Continuous_interest_rate_after_tax: float = 0.780984850207341
    Operating_Expense_Rate: float = 0.000125696868875964
    Research_and_development_expense_rate: float = 0
    Cash_flow_rate: float = 0.458143143520965
    Interest_bearing_debt_interest_rate: float = 0.000725072507250725
    Tax_rate_A: float = 0
    Net_Value_Per_Share_B: float = 0.147949938898487
    Net_Value_Per_Share_A: float = 0.147949938898487
    Net_Value_Per_Share_C: float = 0.147949938898487
    Persistent_EPS_Last_Four_Seasons: float = 0.16914058806845
    Cash_Flow_Per_Share: float = 0.311664426681757
    Revenue_Per_Share: float = 0.0175597803893099
    Operating_Profit_Per_Share: float = 0.0959205276443286
    Per_Share_Net_profit_before_tax: float = 0.138736160277171
    Realized_Sales_Gross_Profit_Growth_Rate: float = 0.0221022783580128
    Operating_Profit_Growth_Rate: float = 0.848194994526472
    After_tax_Net_Profit_Growth_Rate: float = 0.688979462807371
    Regular_Net_Profit_Growth_Rate: float = 0.688979462807371
    Continuous_Net_Profit_Growth_Rate: float = 0.217535386199635
    Total_Asset_Growth_Rate: float = 4980000000
    Net_Value_Growth_Rate: float = 0.000326977269203033
    Total_Asset_Return_Growth_Rate_Ratio: float = 0.263099983681843
    Cash_Reinvestment_percent: float = 0.363725271040572
    Current_Ratio: float = 0.00225896331056689
    Quick_Ratio: float = 0.00120775508523532
    Interest_Expense_Ratio: float = 0.629951302036958
    Total_debt_Total_net_worth: float = 0.0212659243655332
    Debt_ratio_percent: float = 0.207576261450555
    Net_worth_Assets: float = 0.792423738549445
    Long_term_fund_suitability_ratio_A: float = 0.00502445472861451
    Borrowing_dependency: float = 0.390284354359258
    Contingent_liabilities_Net_worth: float = 0.00647850248610705
    Operating_profit_Paid_in_capital: float = 0.0958848339765825
    Net_profit_before_tax_Paid_in_capital: float = 0.137757333534424
    Inventory_and_accounts_receivable_Net_value: float = 0.398035698256887
    Total_Asset_Turnover: float = 0.0869565217391304
    Accounts_Receivable_Turnover: float = 0.00181388412648494
    Average_Collection_Days: float = 0.0034873642818412
    Inventory_Turnover_Rate: float = 0.000182092597429571
    Fixed_Assets_Turnover_Frequency: float = 0.000116500653235806
    Net_Worth_Turnover_Rate: float = 0.0329032258064516
    Revenue_per_person: float = 0.0341641819543792
    Operating_profit_per_person: float = 0.392912869451166
    Allocation_rate_per_person: float = 0.0371353015800987
    Working_Capital_to_Total_Assets: float = 0.67277529248986
    Quick_Assets_Total_Assets: float = 0.166672958825266
    Current_Assets_Total_Assets: float = 0.190642959052727
    Cash_Total_Assets: float = 0.00409440595228806
    Quick_Assets_Current_Liability: float = 0.00199677086064508
    Cash_Current_Liability: float = 0.00014733602476056
    Current_Liability_to_Assets: float = 0.147308450425486
    Operating_Funds_to_Liability: float = 0.334015171333379
    Inventory_Working_Capital: float = 0.276920158240506
    Inventory_Current_Liability: float = 0.00103598999165807
    Current_Liabilities_Liability: float = 0.676269176153092
    Working_Capital_Equity: float = 0.721274551521743
    Current_Liabilities_Equity: float = 0.339077006789355
    Long_term_Liability_Current_Assets: float = 0.02559236799775
    Retained_Earnings_Total_Assets: float = 0.903224771166726
    Total_income_Total_expense: float = 0.00202161301202566
    Total_expense_Assets: float = 0.064855707690831
    Current_Asset_Turnover_Rate: float = 701000000
    Quick_Asset_Turnover_Rate: float = 6550000000
    Working_capitcal_Turnover_Rate: float = 0.593830503987655
    Cash_Turnover_Rate: float = 458000000
    Cash_Flow_to_Sales: float = 0.6715676535815
    Fixed_Assets_to_Assets: float = 0.424205762216667
    Current_Liability_to_Liability: float = 0.676269176153092
    Current_Liability_to_Equity: float = 0.339077006789355
    Equity_to_Long_term_Liability: float = 0.126549487816618
    Cash_Flow_to_Total_Assets: float = 0.637555395323871
    Cash_Flow_to_Liability: float = 0.458609147666847
    CFO_to_Assets: float = 0.52038191789012
    Cash_Flow_to_Equity: float = 0.312904948119326
    Current_Liability_to_Current_Assets: float = 0.11825047660899
    Liability_Assets_Flag: float = 0
    Net_Income_to_Total_Assets: float = 0.716845343217827
    Total_assets_to_GNP_price: float = 0.00921944002110296
    No_credit_Interval: float = 0.622878959445127
    Gross_Profit_to_Sales: float = 0.601453290101533
    Net_Income_to_Stockholder_Equity: float = 0.82789021403512
    Liability_to_Equity: float = 0.29020189277926
    Degree_of_Financial_Leverage: float = 0.0266006307607414
    Interest_Coverage_Ratio: float = 0.564050112276341
    Net_Income_Flag: float = 1
    Equity_to_Liability: float = 0.0164687409123162

class OutputData(BaseModel):
    score: float

@app.post("/score", response_model=OutputData)
def score(data: InputData):
    model_input = np.array(list(data.dict().values())).reshape(1, -1)
    result = float(model.predict_proba(model_input)[:, -1][0])
    return {"score": result}
