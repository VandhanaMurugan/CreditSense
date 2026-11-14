"""FastAPI service.  Run:  uvicorn creditsense.api:app --reload   (needs artifacts/creditsense_bundle.joblib; build with `make bundle`)"""
import io
from typing import Optional
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .serve import CreditSense

app = FastAPI(title="CreditSense API", version="2.0.0", description="Calibrated, cost-aware and guaranteed credit decisions (decision support).")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_cs = None
def cs():
    global _cs
    if _cs is None: _cs = CreditSense()
    return _cs


class Applicant(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, description="Revolving balance / credit limits")
    age: float = Field(..., ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    model_config = {"populate_by_name": True}


class Policy(BaseModel):
    r: Optional[float] = Field(None, gt=0, description="cost of approving a defaulter / cost of declining a good borrower")
    c: Optional[float] = Field(None, ge=0, description="cost of one analyst referral, same units")
    rho: Optional[float] = Field(None, ge=0, le=1, description="referral budget (share of applicants); null = unlimited")
    alpha: Optional[float] = Field(None, ge=0, le=1, description="cap on share of defaulters auto-approved; 0 = no cap")
    delta: Optional[float] = Field(None, gt=0, lt=1, description="set 0.05 for a 95%-confidence (PAC) cap")


class ScoreRequest(BaseModel):
    applicant: Applicant
    policy: Optional[Policy] = None


def _pol(p: Optional[Policy]): return {k: v for k, v in (p.model_dump() if p else {}).items() if v is not None}


@app.get("/health")
def health(): return {"status": "ok", "model_version": cs().b["version"]}


@app.post("/score")
def score(req: ScoreRequest):
    try: return cs().score(req.applicant.model_dump(by_alias=True), _pol(req.policy))
    except ValueError as e: raise HTTPException(422, str(e))


@app.post("/score_batch")
async def score_batch(file: UploadFile = File(...), r: Optional[float] = None, c: Optional[float] = None,
                      rho: Optional[float] = None, alpha: Optional[float] = None):
    df = pd.read_csv(io.BytesIO(await file.read()))
    out = cs().score_batch(df, {k: v for k, v in dict(r=r, c=c, rho=rho, alpha=alpha).items() if v is not None})
    buf = io.StringIO(); out.to_csv(buf, index=False); buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=scored.csv"})


@app.get("/policy_simulation")
def policy_simulation(r: float = 5, c: float = 0.2, rho: Optional[float] = 0.2, alpha: float = 0.10, delta: Optional[float] = None):
    return cs().policy_simulation(r=r, c=c, rho=rho, alpha=alpha, delta=delta)


@app.post("/monitoring")
async def monitoring(file: UploadFile = File(...), label_column: str = "SeriousDlqin2yrs"):
    df = pd.read_csv(io.BytesIO(await file.read()))
    y = df.pop(label_column).values if label_column in df else None
    return cs().monitoring(df, y)


@app.get("/model_card")
def model_card(): return cs().model_card()
