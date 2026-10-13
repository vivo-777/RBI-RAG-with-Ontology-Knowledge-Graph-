from pydantic import BaseModel, Field
from typing import List, Optional

class RegulatedEntity(BaseModel):
    name: str
    type: str  # Bank, NBFC, Cooperative Bank, SFB, Payment Aggregator

class Department(BaseModel):
    name: str  # DoR, DPSS, FED, FIDD
    full_name: Optional[str] = None

class RegulatoryConcept(BaseModel):
    name: str  # KYC, Basel III, CRR, NPA Classification

class Circular(BaseModel):
    document_id: str
    filename: str
    date: str
    department: Optional[Department] = None
    status: str  # active, superseded, amended
    supersedes: List[str] = Field(default_factory=list) # List of document_ids this circular supersedes
    amends: List[str] = Field(default_factory=list)     # List of document_ids this circular amends
    applies_to: List[RegulatedEntity] = Field(default_factory=list)

class Obligation(BaseModel):
    id: str
    description: str
    derived_from: str # document_id
