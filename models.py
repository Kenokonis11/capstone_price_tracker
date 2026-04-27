from sqlalchemy import Column, Float, String

from database import Base


class DBAsset(Base):
    __tablename__ = "assets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    current_value = Column(Float, nullable=False, default=0.0)
    state_json = Column(String, nullable=False)
