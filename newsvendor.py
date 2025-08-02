import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class NewsvendorEnvironment:
    def __init__(self, purchase_price=0.5, selling_price=1.0, salvage_value=0.2,
                 demand_mean=100, demand_std=20, random_seed=None):
        """
        Initialize the newsvendor environment.
        EX NO: 1
        Date PROGRAM TO SIMULATE A NEWSVENDOR AGENT

        Parameters:
        - purchase_price: Cost to buy one newspaper
        - selling_price: Revenue from selling one newspaper
        - salvage_value: Value of unsold newspapers
        - demand_mean: Average daily demand
        - demand_std: Standard deviation of daily demand
        - random_seed: Seed for reproducibility
        """
        self.purchase_price = purchase_price
        self.selling_price = selling_price
        self.salvage_value = salvage_value
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.rng = np.random.default_rng(random_seed)

        # Track history
        self.history = {
            'demand': [],
            'order': [],
            'profit': [],
            'lost_sales': [],
            'waste': []
        }

    def generate_demand(self):
        """Generate random demand for the day"""
        demand = self.rng.normal(self.demand_mean, self.demand_std)
        return max(0, int(round(demand)))

    def step(self, order_quantity):
        """
        Simulate one day of operation.

        Parameters:
        - order_quantity: Number of newspapers purchased

        Returns:
        - profit: Daily profit
        - lost_sales: Demand not met
        - waste: Unsold newspapers
        """
        demand = self.generate_demand()
        sales = min(demand, order_quantity)
        lost_sales = max(0, demand - order_quantity)
        waste = max(0, order_quantity - demand)
        revenue = sales * self.selling_price
        cost = order_quantity * self.purchase_price
        salvage = waste * self.salvage_value
        profit = revenue - cost + salvage

        # Update history
        self.history['demand'].append(demand)
        self.history['order'].append(order_quantity)
        self.history['profit'].append(profit)
        self.history['lost_sales'].append(lost_sales)
        self.history['waste'].append(waste)

        return profit, lost_sales, waste

    def get_history(self):
        """Return the simulation history"""
        return self.history

class NewsvendorAgent:
    def __init__(self, env):
        """Initialize the agent with the environment"""
        self.env = env
        self.observed_demand = []

    def observe(self, demand):
        """Record observed demand"""
        self.observed_demand.append(demand)

    def calculate_order_quantity(self):
        """
        Calculate optimal order quantity using Newsvendor formula.

        Returns:
        - order_quantity: Optimal number of newspapers to order
        """
        if not self.observed_demand:
            # If no history, make a guess (could be improved)
            return int(self.env.demand_mean)

        # Calculate critical ratio
        underage_cost = self.env.selling_price - self.env.purchase_price
        overage_cost = self.env.purchase_price - self.env.salvage_value
        critical_ratio = underage_cost / (underage_cost + overage_cost)

        # Estimate demand distribution parameters from observed data
        mean = np.mean(self.observed_demand)
        std = np.std(self.observed_demand)

        # Handle case where std is 0 (all demands are same)
        if std == 0:
            return int(mean)

        # Calculate optimal order quantity using inverse CDF
        order_quantity = norm.ppf(critical_ratio, loc=mean, scale=std)
        return max(0, int(round(order_quantity)))

def simulate(days=30, warmup_days=7, demand_mean=100, demand_std=20,
             purchase_price=0.5, selling_price=1.0, salvage_value=0.2):
    """
    Run the newsvendor simulation.

    Parameters:
    - days: Total number of days to simulate
    - warmup_days: Days before agent starts making decisions
    - demand_mean: Average daily demand
    - demand_std: Standard deviation of daily demand
    - purchase_price: Cost to buy one newspaper
    - selling_price: Revenue from selling one newspaper
    - salvage_value: Value of unsold newspapers
    """
    # Initialize environment and agent
    env = NewsvendorEnvironment(
        purchase_price=purchase_price,
        selling_price=selling_price,
        salvage_value=salvage_value,
        demand_mean=demand_mean,
        demand_std=demand_std
    )
    agent = NewsvendorAgent(env)

    # Run simulation
    for day in range(days):
        if day < warmup_days:
            # Initial warmup period - order average demand
            order_quantity = demand_mean
        else:
            # Agent calculates order quantity based on observed demand
            order_quantity = agent.calculate_order_quantity()

        # Simulate one day
        profit, lost_sales, waste = env.step(order_quantity)

        # Agent observes the actual demand
        agent.observe(env.history['demand'][-1])

        # Print daily results
        print(f"Day {day+1}: Ordered {order_quantity}, Demand {env.history['demand'][-1]}, "
              f"Profit ${profit:.2f}, Lost sales {lost_sales}, Waste {waste}")

    # Print summary statistics
    history = env.get_history()
    total_profit = sum(history['profit'])
    avg_profit = total_profit / days
    total_lost_sales = sum(history['lost_sales'])
    total_waste = sum(history['waste'])

    print("\nSimulation Summary:")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Average daily profit: ${avg_profit:.2f}")
    print(f"Total lost sales: {total_lost_sales} newspapers")
    print(f"Total waste: {total_waste} newspapers")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(history['demand'], label='Demand', marker='o')
    plt.plot(history['order'], label='Order', marker='x')
    plt.title('Newsvendor Simulation: Demand vs Order Quantity')
    plt.xlabel('Day')
    plt.ylabel('Newspapers')
    plt.legend()
    plt.grid(True)
    plt.show()

    return history

# Run the simulation
if __name__ == "__main__":
    history = simulate(days=30, warmup_days=7, demand_mean=100, demand_std=20)
