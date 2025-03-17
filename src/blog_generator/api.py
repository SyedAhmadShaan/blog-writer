from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from blog_generator.crew import BlogGenerator
from typing import Optional, List, Dict, Any
from datetime import datetime
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Blog Generator API",
    description="API endpoint for running the Blog Generator crew to create blog content.",
    version="0.1.0",
)


class CrewInput(BaseModel):
    topic: str = Field(..., min_length=1, max_length=1000)
    current_year: Optional[int] = Field(default_factory=lambda: datetime.now().year)


class TaskOutput(BaseModel):
    task_name: str
    agent_name: str
    message: str


class AgentOutput(BaseModel):
    name: str
    role: str
    goal: str
    backstory: str


class TokenUsage(BaseModel):
    total_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    cached_prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    successful_requests: Optional[int] = None


class CrewOutput(BaseModel):
    raw_output: str
    tasks: List[TaskOutput]
    agents: List[AgentOutput]
    token_usage: Optional[TokenUsage] = None


class CrewResponse(BaseModel):
    status: str
    output: Optional[CrewOutput] = None
    error_details: Optional[str] = None


@app.post("/run", response_model=CrewResponse, summary="Run the Blog Generator Crew")
async def run_crew(input_data: CrewInput):
    """
    Run the Blog Generator crew with the provided topic and context.

    This endpoint initializes the BlogGenerator crew and kicks off the content generation process.
    """
    try:
        logger.info(f"Received input: {input_data}")
        blog_crew = BlogGenerator()
        crew = blog_crew.crew()
        output = crew.kickoff(inputs=input_data.model_dump())
        logger.info(f"Crew execution successful. Output type: {type(output)}")

        # Process Task Outputs
        task_outputs = []
        if isinstance(output, dict) and "tasks_output" in output:
            for task in output["tasks_output"]:
                task_outputs.append(
                    TaskOutput(
                        task_name=task.get("task_name", "Unknown Task"),
                        agent_name=task.get("agent_name", "Unknown Agent"),
                        message=task.get("message", "No response"),
                    )
                )

        # Process Agent Information
        agent_outputs = []
        for agent in crew.agents:
            agent_outputs.append(
                AgentOutput(
                    name=agent.role,
                    role=agent.role,
                    goal=agent.goal,
                    backstory=agent.backstory,
                )
            )

        # Build Final Crew Output
        crew_output = CrewOutput(
            raw_output=(
                output.get("raw", str(output))
                if isinstance(output, dict)
                else str(output)
            ),
            tasks=task_outputs,
            agents=agent_outputs,
        )

        return CrewResponse(status="success", output=crew_output)

    except Exception as e:
        logger.error(f"Error in run_crew: {str(e)}")
        logger.error(traceback.format_exc())
        return CrewResponse(status="error", error_details=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
