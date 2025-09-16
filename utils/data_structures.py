from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class Skill(BaseModel):
    name: str = Field(description="Name of the skill")
    description: Optional[str] = Field(description="Description of the skill")
    category: str = Field(description="Category of the skill (technical or soft)")
    level: Optional[int] = None
    confidence: float = Field(description="Confidence in the skill assessment (0-1)")
    evidence: str = Field(description="Evidence supporting the skill assessment")
    years_experience: Optional[float] = None
    level_basis: str = Field(description="Basis for determining the skill level (years or explicit)")

    # # Custom validator here
    # @validator('level', 'years_experience', pre=True)
    # def allow_none(cls, v):
    #     if v is None:
    #          return None
    #     else:
    #         return v

# Define the output structure
class SkillsOutput(BaseModel):
    skills: List[Skill] = Field(description="List of extracted skills")
# Create the parser

class MatchOutput(BaseModel):
    required_skills: list[str] = Field(..., description="List of required skills for the position")
    candidate_skills: list[str] = Field(..., description="List of skills possessed by the candidate")
    matching_skills: list[str] = Field(..., description="List of common (matching) skills")
    match_score: int = Field(..., description="Match score (a value between 0-100)")
    justification: str = Field(..., description="Short justification summarizing why this score was assigned")

class Concept(BaseModel):
    name: str = Field(description="Name of the concept")
    level: str = Field(description="Level of the concept (Beginner, Intermediate, Advanced)")
    description: Optional[str] = Field(description="Description of the concept")
    requires: Optional[List[str]] = Field(description="List of prerequisite concepts")

class SkillHierarchyOutput(BaseModel):
    skill: str = Field(description="The main skill being analyzed")
    concepts: List[Concept] = Field(description="A list of concepts needed to learn the skill")
    justification: str = Field(..., description="Short justification summarizing why this score was assigned")

# Define Pydantic model for course recommendation
class CourseRecommendation(BaseModel):
    title: str = Field(description="The title of the course")
    provider: str = Field(description="The provider or institution offering the course")
    duration: str = Field(description="The duration of the course")
    summary: str = Field(description="A brief summary of the course content")
    link: str = Field(description="URL link to the course")
    level: Optional[str] = Field(None, description="The difficulty level of the course (Beginner, Intermediate, Advanced)")
    position: Optional[int] = Field(None, description="The position of the course in the learning path")
    prerequisites: Optional[List[str]] = Field(default_factory=list, description="List of courses that should be completed before this one")
    builds_toward: Optional[List[str]] = Field(default_factory=list, description="List of courses that this course builds toward")

class CourseRecommendations(BaseModel):
    courses: List[CourseRecommendation] = Field(description="List of recommended courses")

# Define Pydantic model for learning path recommendation
class LearningPathRecommendation(BaseModel):
    path_name: str = Field(description="The name of the recommended learning or career path")
    path_type: str = Field(description="The type of path (learning_path or career_path)")
    description: Optional[str] = Field(None, description="A brief description of the path")
    reason: str = Field(description="Reason why this path was recommended based on the query")