
import modal

stub = modal.App("debug-safety-gym-structure")

image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "libsdl2-dev", 
    "libsdl2-image-dev", 
    "libsdl2-mixer-dev", 
    "libsdl2-ttf-dev", 
    "libfreetype6-dev", 
    "libportmidi-dev", 
    "libjpeg-dev", 
    "pkg-config",
    "git",
    "build-essential"
).pip_install("safety-gymnasium", "gymnasium")

@stub.function(image=image)
def inspect_env():
    import safety_gymnasium
    import gymnasium as gym
    
    print("Inspecting SafetyPointGoal1-v0 structure...")
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    
    print("Calling env.reset()...")
    env.reset()
    
    print(f"Env type: {type(env)}")
    print(f"Unwrapped type: {type(env.unwrapped)}")
    
    # Check for task
    if hasattr(env.unwrapped, 'task'):
        print("Found env.unwrapped.task")
        task = env.unwrapped.task
        print(f"Task type: {type(task)}")
        print(f"Task dir: {dir(task)}")
        
        if hasattr(task, 'hazards'):
            print("Found task.hazards")
            print(f"Hazards type: {type(task.hazards)}")
            print(f"Hazards dir: {dir(task.hazards)}")
            if hasattr(task.hazards, 'pos'):
                print(f"Hazards pos: {task.hazards.pos}")
            elif hasattr(task.hazards, 'locations'):
                 print(f"Hazards locations: {task.hazards.locations}")
        else:
            print("No task.hazards found")
            
        if hasattr(task, 'obstacles'):
             print("Found task.obstacles")
             print(f"Obstacles dir: {dir(task.obstacles)}")
    else:
        print("No env.unwrapped.task found")
        # Try direct attributes
        print(f"Unwrapped dir: {dir(env.unwrapped)}")

