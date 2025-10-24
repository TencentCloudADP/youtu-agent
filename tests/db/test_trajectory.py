# from utu.agents.common import TaskRecorder
from utu.agents import SimpleAgent
from utu.db import DBService, TrajectoryModel


async def test_traj_model():
    """Test TrajectoryModel. The recorded trajectory should be saved to db and can be visualized."""
    agent = SimpleAgent(config="simple/base")
    task_recorder = await agent.run("hello")
    trajectory = TrajectoryModel.from_task_recorder(task_recorder)
    DBService.add(trajectory)
