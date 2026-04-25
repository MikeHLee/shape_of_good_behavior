"""
Coding Scenario Generator

Generates programming problem-solving scenarios for training:
- Bug fixing
- Code completion
- Algorithm design
- Refactoring
- Code review

Each scenario presents a coding context and multiple possible solutions,
with assessments of correctness, efficiency, style, and safety.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import random


@dataclass
class CodeSolution:
    """A candidate solution to a coding problem."""
    code: str
    description: str
    correctness: float  # 0-1, does it work?
    efficiency: float  # 0-1, is it performant?
    readability: float  # 0-1, is it clean?
    safety: float  # 0-1, does it have security issues?
    explanation: str


@dataclass
class CodingScenario:
    """A coding problem with multiple solution approaches."""
    scenario_id: str
    category: str  # bugfix, completion, algorithm, refactor, review
    context: str  # Background information
    problem_code: str  # The code to work with
    problem_description: str
    solutions: List[CodeSolution]
    constraints: List[str] = field(default_factory=list)  # Style/approach constraints
    language: str = "python"


@dataclass
class CodingTransition:
    """A state-action-result triple for coding training."""
    state: str  # Problem context
    action: str  # Chosen solution
    result: str  # Outcome description
    reward: float  # Combined quality score
    cost: float  # Penalty for bugs/security issues
    category: str


class CodingScenarioGenerator:
    """
    Generates diverse coding scenarios for training.
    
    Categories:
    1. BUGFIX: Find and fix bugs in code
    2. COMPLETION: Complete partial implementations
    3. ALGORITHM: Design algorithms for given problems
    4. REFACTOR: Improve existing code structure
    5. REVIEW: Identify issues in code reviews
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.scenarios = self._build_scenario_library()
    
    def _build_scenario_library(self) -> List[CodingScenario]:
        """Build library of coding scenarios."""
        scenarios = []
        
        # === BUGFIX SCENARIOS ===
        scenarios.append(CodingScenario(
            scenario_id="bugfix_001",
            category="bugfix",
            context="A junior developer wrote this function but it's returning None unexpectedly.",
            problem_code='''def find_max(numbers):
    """Find the maximum value in a list."""
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    # Bug: missing return statement''',
            problem_description="The function doesn't return the maximum value.",
            solutions=[
                CodeSolution(
                    code='''def find_max(numbers):
    """Find the maximum value in a list."""
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val''',
                    description="Add the missing return statement",
                    correctness=1.0,
                    efficiency=0.8,
                    readability=0.9,
                    safety=0.9,
                    explanation="Simple fix - just needed the return statement",
                ),
                CodeSolution(
                    code='''def find_max(numbers):
    """Find the maximum value in a list."""
    return max(numbers)''',
                    description="Use built-in max function",
                    correctness=1.0,
                    efficiency=1.0,
                    readability=1.0,
                    safety=0.9,
                    explanation="More Pythonic, uses built-in for better performance",
                ),
                CodeSolution(
                    code='''def find_max(numbers):
    """Find the maximum value in a list."""
    numbers.sort()
    return numbers[-1]''',
                    description="Sort and return last element",
                    correctness=0.8,  # Mutates input
                    efficiency=0.3,  # O(n log n) instead of O(n)
                    readability=0.7,
                    safety=0.6,  # Side effect of mutation
                    explanation="Works but inefficient and mutates input",
                ),
            ],
        ))
        
        scenarios.append(CodingScenario(
            scenario_id="bugfix_002",
            category="bugfix",
            context="This binary search implementation has an off-by-one error.",
            problem_code='''def binary_search(arr, target):
    left, right = 0, len(arr)  # Bug: should be len(arr) - 1
    while left < right:  # Bug: should be left <= right
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid  # Bug: should be mid + 1
        else:
            right = mid  # Bug: should be mid - 1
    return -1''',
            problem_description="Binary search doesn't find elements correctly and may infinite loop.",
            solutions=[
                CodeSolution(
                    code='''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',
                    description="Fix all boundary conditions",
                    correctness=1.0,
                    efficiency=1.0,
                    readability=0.9,
                    safety=1.0,
                    explanation="Correct binary search with proper boundaries",
                ),
                CodeSolution(
                    code='''def binary_search(arr, target):
    import bisect
    idx = bisect.bisect_left(arr, target)
    if idx < len(arr) and arr[idx] == target:
        return idx
    return -1''',
                    description="Use bisect module",
                    correctness=1.0,
                    efficiency=1.0,
                    readability=0.8,
                    safety=1.0,
                    explanation="Uses standard library - reliable but less educational",
                ),
            ],
        ))
        
        # === COMPLETION SCENARIOS ===
        scenarios.append(CodingScenario(
            scenario_id="completion_001",
            category="completion",
            context="Complete the implementation of a simple LRU cache.",
            problem_code='''class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        # TODO: Initialize data structures
        
    def get(self, key: int) -> int:
        # TODO: Return value if exists, -1 otherwise
        # TODO: Update access order
        pass
        
    def put(self, key: int, value: int) -> None:
        # TODO: Add/update key-value pair
        # TODO: Evict LRU item if at capacity
        pass''',
            problem_description="Implement an LRU cache with O(1) get and put operations.",
            solutions=[
                CodeSolution(
                    code='''from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
        
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)''',
                    description="Use OrderedDict for O(1) operations",
                    correctness=1.0,
                    efficiency=1.0,
                    readability=0.95,
                    safety=1.0,
                    explanation="Elegant solution using Python's OrderedDict",
                ),
                CodeSolution(
                    code='''class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []
        
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]
        
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            del self.cache[self.order.pop(0)]
        self.cache[key] = value
        self.order.append(key)''',
                    description="Use list for order tracking",
                    correctness=1.0,
                    efficiency=0.4,  # O(n) remove operations
                    readability=0.8,
                    safety=0.9,
                    explanation="Works but O(n) for get/put due to list operations",
                ),
            ],
        ))
        
        # === ALGORITHM SCENARIOS ===
        scenarios.append(CodingScenario(
            scenario_id="algorithm_001",
            category="algorithm",
            context="Design an algorithm to find all anagram groups in a list of words.",
            problem_code='''# Input: ["eat", "tea", "tan", "ate", "nat", "bat"]
# Output: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]

def group_anagrams(words):
    # TODO: Implement algorithm
    pass''',
            problem_description="Group words that are anagrams of each other.",
            solutions=[
                CodeSolution(
                    code='''def group_anagrams(words):
    groups = {}
    for word in words:
        key = ''.join(sorted(word))
        if key not in groups:
            groups[key] = []
        groups[key].append(word)
    return list(groups.values())''',
                    description="Sort characters as key",
                    correctness=1.0,
                    efficiency=0.8,  # O(n * k log k) where k is word length
                    readability=0.9,
                    safety=1.0,
                    explanation="Clean solution using sorted characters as dictionary key",
                ),
                CodeSolution(
                    code='''from collections import defaultdict

def group_anagrams(words):
    groups = defaultdict(list)
    for word in words:
        # Use character count tuple as key
        count = [0] * 26
        for c in word:
            count[ord(c) - ord('a')] += 1
        groups[tuple(count)].append(word)
    return list(groups.values())''',
                    description="Character count as key (O(n*k))",
                    correctness=1.0,
                    efficiency=1.0,  # O(n * k)
                    readability=0.7,
                    safety=0.9,  # Assumes lowercase letters only
                    explanation="More efficient but less readable",
                ),
            ],
        ))
        
        # === REFACTOR SCENARIOS ===
        scenarios.append(CodingScenario(
            scenario_id="refactor_001",
            category="refactor",
            context="This code works but has poor structure. Refactor for maintainability.",
            problem_code='''def process(data):
    result = []
    for item in data:
        if item['type'] == 'A':
            if item['value'] > 100:
                if item['status'] == 'active':
                    result.append(item['value'] * 2)
                else:
                    result.append(item['value'])
            else:
                if item['status'] == 'active':
                    result.append(item['value'] * 1.5)
                else:
                    result.append(item['value'] * 0.5)
        elif item['type'] == 'B':
            if item['value'] > 50:
                result.append(item['value'] + 10)
            else:
                result.append(item['value'])
    return result''',
            problem_description="Deeply nested conditionals make this hard to maintain.",
            solutions=[
                CodeSolution(
                    code='''def process_type_a(item):
    multiplier = 1.0
    if item['status'] == 'active':
        multiplier = 2.0 if item['value'] > 100 else 1.5
    else:
        multiplier = 1.0 if item['value'] > 100 else 0.5
    return item['value'] * multiplier

def process_type_b(item):
    return item['value'] + 10 if item['value'] > 50 else item['value']

def process(data):
    processors = {'A': process_type_a, 'B': process_type_b}
    return [
        processors[item['type']](item)
        for item in data
        if item['type'] in processors
    ]''',
                    description="Extract functions, use dispatch table",
                    correctness=1.0,
                    efficiency=0.9,
                    readability=1.0,
                    safety=0.9,
                    explanation="Clear separation of concerns, easy to extend",
                ),
                CodeSolution(
                    code='''def process(data):
    result = []
    for item in data:
        value = item['value']
        is_active = item['status'] == 'active'
        
        if item['type'] == 'A':
            high_value = value > 100
            if high_value and is_active:
                result.append(value * 2)
            elif high_value:
                result.append(value)
            elif is_active:
                result.append(value * 1.5)
            else:
                result.append(value * 0.5)
        elif item['type'] == 'B':
            result.append(value + 10 if value > 50 else value)
    return result''',
                    description="Extract variables, flatten slightly",
                    correctness=1.0,
                    efficiency=0.9,
                    readability=0.7,
                    safety=0.9,
                    explanation="Improved but still has nested conditionals",
                ),
            ],
        ))
        
        # === REVIEW SCENARIOS ===
        scenarios.append(CodingScenario(
            scenario_id="review_001",
            category="review",
            context="Code review: Identify security and correctness issues.",
            problem_code='''import sqlite3
import os

def get_user(user_id):
    conn = sqlite3.connect('users.db')
    # SQL Injection vulnerability!
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = conn.execute(query).fetchone()
    conn.close()
    return result

def run_command(cmd):
    # Command injection vulnerability!
    os.system(f"echo {cmd}")
    
def read_file(filename):
    # Path traversal vulnerability!
    with open(f"/data/{filename}") as f:
        return f.read()''',
            problem_description="Find security vulnerabilities in this code.",
            solutions=[
                CodeSolution(
                    code='''import sqlite3
import subprocess
import os

def get_user(user_id):
    conn = sqlite3.connect('users.db')
    # Use parameterized query
    query = "SELECT * FROM users WHERE id = ?"
    result = conn.execute(query, (user_id,)).fetchone()
    conn.close()
    return result

def run_command(cmd):
    # Use subprocess with shell=False
    subprocess.run(["echo", cmd], check=True)
    
def read_file(filename):
    # Validate path
    base = "/data"
    filepath = os.path.normpath(os.path.join(base, filename))
    if not filepath.startswith(base):
        raise ValueError("Invalid path")
    with open(filepath) as f:
        return f.read()''',
                    description="Fix all security vulnerabilities",
                    correctness=1.0,
                    efficiency=0.9,
                    readability=0.9,
                    safety=1.0,
                    explanation="Parameterized queries, subprocess, path validation",
                ),
                CodeSolution(
                    code='''# Only fix SQL injection
def get_user(user_id):
    conn = sqlite3.connect('users.db')
    query = "SELECT * FROM users WHERE id = ?"
    result = conn.execute(query, (user_id,)).fetchone()
    conn.close()
    return result''',
                    description="Partial fix - only SQL injection",
                    correctness=0.5,
                    efficiency=0.9,
                    readability=0.9,
                    safety=0.4,  # Other vulns remain
                    explanation="Incomplete - other vulnerabilities not addressed",
                ),
            ],
        ))
        
        return scenarios
    
    def generate_episode(
        self,
        num_problems: int = 5,
        categories: Optional[List[str]] = None,
    ) -> List[CodingTransition]:
        """
        Generate an episode of coding problems.
        
        Args:
            num_problems: Number of problems in the episode
            categories: Filter to specific categories
        
        Returns:
            List of transitions for training
        """
        if categories:
            pool = [s for s in self.scenarios if s.category in categories]
        else:
            pool = self.scenarios
        
        if not pool:
            pool = self.scenarios
        
        episode = []
        selected = random.choices(pool, k=num_problems)
        
        for scenario in selected:
            # Choose a solution (biased toward better ones)
            solutions = scenario.solutions
            weights = [
                s.correctness * 0.4 + s.efficiency * 0.2 + 
                s.readability * 0.2 + s.safety * 0.2 + 0.1
                for s in solutions
            ]
            solution = random.choices(solutions, weights=weights, k=1)[0]
            
            state = f"""Category: {scenario.category}
Context: {scenario.context}

Code:
```{scenario.language}
{scenario.problem_code}
```

Problem: {scenario.problem_description}"""
            
            # Compute reward
            reward = (
                solution.correctness * 0.4 +
                solution.efficiency * 0.2 +
                solution.readability * 0.2 +
                solution.safety * 0.2
            )
            
            # Cost for low safety
            cost = max(0, 0.5 - solution.safety) * 2
            
            episode.append(CodingTransition(
                state=state,
                action=f"Solution: {solution.description}\n\n```{scenario.language}\n{solution.code}\n```",
                result=solution.explanation,
                reward=reward,
                cost=cost,
                category=scenario.category,
            ))
        
        return episode
    
    def generate_preference_pairs(
        self,
        num_pairs: int = 10,
    ) -> List[Dict[str, Any]]:
        """Generate preference pairs for training."""
        pairs = []
        
        for scenario in random.choices(self.scenarios, k=num_pairs):
            if len(scenario.solutions) < 2:
                continue
            
            # Sort by overall quality
            def quality(s):
                return s.correctness * 0.4 + s.efficiency * 0.2 + s.readability * 0.2 + s.safety * 0.2
            
            sorted_solutions = sorted(scenario.solutions, key=quality, reverse=True)
            
            pairs.append({
                "context": f"Problem: {scenario.problem_description}\n\nCode:\n{scenario.problem_code}",
                "preferred": sorted_solutions[0].code,
                "dispreferred": sorted_solutions[-1].code,
                "category": scenario.category,
            })
        
        return pairs
    
    def to_training_data(self, num_episodes: int = 50) -> Dict[str, Any]:
        """Generate complete training dataset."""
        episodes = []
        for _ in range(num_episodes):
            episode = self.generate_episode(num_problems=random.randint(2, 5))
            episodes.append([
                {
                    "state": t.state,
                    "action": t.action,
                    "result": t.result,
                    "reward": t.reward,
                    "cost": t.cost,
                    "category": t.category,
                }
                for t in episode
            ])
        
        return {
            "episodes": episodes,
            "preference_pairs": self.generate_preference_pairs(num_episodes),
            "metadata": {
                "categories": ["bugfix", "completion", "algorithm", "refactor", "review"],
                "language": "python",
            },
        }


if __name__ == "__main__":
    print("=== Coding Scenario Generator Demo ===\n")
    
    generator = CodingScenarioGenerator()
    
    episode = generator.generate_episode(num_problems=2)
    for i, t in enumerate(episode):
        print(f"\n--- Problem {i+1} ({t.category}) ---")
        print(f"Reward: {t.reward:.2f}, Cost: {t.cost:.2f}")
        print(f"Result: {t.result}")
    
    print("\n\n=== Preference Pairs ===")
    pairs = generator.generate_preference_pairs(num_pairs=1)
    for pair in pairs:
        print(f"\nCategory: {pair['category']}")
        print(f"Preferred solution length: {len(pair['preferred'])} chars")
