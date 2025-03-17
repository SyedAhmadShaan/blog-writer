from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class BlogGenerator:
    """
    BlogGenerator Crew

    This crew orchestrates the process of generating engaging blog content by coordinating three roles:
    - Content Planner: Develops detailed content outlines.
    - Content Writer: Crafts comprehensive blog posts.
    - Content Editor: Refines and polishes the final content.

    YAML configuration files for agents and tasks are referenced below.
    """

    # YAML configuration file paths for agents and tasks.
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def content_planner(self) -> Agent:
        """
        Initialize the Content Planner Agent using its YAML configuration.
        """
        return Agent(config=self.agents_config["content_planner"], verbose=True)

    @agent
    def content_writer(self) -> Agent:
        """
        Initialize the Content Writer Agent using its YAML configuration.
        """
        return Agent(config=self.agents_config["content_writer"], verbose=True)

    @agent
    def content_editor(self) -> Agent:
        """
        Initialize the Content Editor Agent using its YAML configuration.
        """
        return Agent(config=self.agents_config["content_editor"], verbose=True)

    @task
    def planning_task(self) -> Task:
        """
        Define the planning task to create a detailed content outline.
        """
        return Task(config=self.tasks_config["planning_task"])

    @task
    def writing_task(self) -> Task:
        """
        Define the writing task to convert the outline into a comprehensive blog post.
        """
        return Task(config=self.tasks_config["writing_task"])

    @task
    def editing_task(self) -> Task:
        """
        Define the editing task to review and refine the blog post,
        outputting a polished report.
        """
        return Task(config=self.tasks_config["editing_task"], output_file="report1.md")

    @crew
    def crew(self) -> Crew:
        """
        Assemble the blog generator crew by linking agents and tasks into a sequential process.
        """
        return Crew(
            agents=self.agents,  # Automatically registered via the @agent decorator.
            tasks=self.tasks,  # Automatically registered via the @task decorator.
            process=Process.sequential,
            verbose=True,
        )
