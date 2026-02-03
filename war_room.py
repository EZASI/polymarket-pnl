import os
import sys

# Simulation of the War Room if dependencies are missing
def simulate_war_room():
    print("### STARTING WAR ROOM DEBATE (SIMULATED) ###\n")
    
    print("\n--- AGENT: AGGRESSIVE QUANT ---")
    print("PROPOSAL: 15m 'Gamma Scalping' on Polymerket")
    print("LOGIC:")
    print("1.  **Entry Trigger**: Monitor the 1-minute order book depth. When the bid-ask spread widens to > 5 cents on a high-volume crypto market (e.g., 'Will BTC hit $100k?'), it indicates temporary dislocation.")
    print("2.  **Volume Spike**: Wait for a trade volume spike > 3x the 20-period moving average. This confirms 'smart money' positioning.")
    print("3.  **Execution**: Immediately place a LIMIT ORDER at the mid-price. DO NOT take liquidity. Force the market to come to you.")
    print("4.  **The Arbitrage**: If YES is $0.40 and NO is $0.55 (Total $0.95), buy BOTH. Instant 5% risk-free yield if filled.")
    
    print("\n--- AGENT: RISK MANAGER ---")
    print("CRITIQUE: You're ignoring the 'Legging Risk'.")
    print("1.  **Fee Drag**: Polymarket fees + gas (if not using proxy) kill the 5-cent spread. You need > 3% net margin.")
    print("2.  **Execution Risk**: You buy YES at $0.40, but before you buy NO at $0.55, it jumps to $0.65. Now you're net long directional with a guaranteed loss if you hedge.")
    print("3.  **Phantom Liquidity**: That 5-cent spread exists because there's no depth. Your 1000-share order will slip the price by 10 cents.")
    
    print("\n--- AGENT: CHIEF STRATEGIST ---")
    print("FINAL VERDICT: The 'Safe' 15m Strategy")
    print("We will not do pure spread scalping. It's too risky with current liquidity.")
    print("INSTEAD, we run the **'Volume-Weighted Convergence'** Strategy:")
    
    print("\n### GOLDEN LOGIC ###")
    print("1.  **Filter**: Only trade markets expiring in < 15 minutes.")
    print("2.  **Signal**: Watch for **Price-Volume Divergence**.")
    print("    - If Price of YES drops 5% but Volume on NO is flat -> False drop (Noise).")
    print("    - **ACTION**: Buy YES (Mean Reversion).")
    print("3.  **Hedge**: Automatically place a 'Take Profit' limit order at +4 cents from entry.")
    print("4.  **Safety**: Max position size = 10% of Average Daily Volume of that specific market.")
    print("5.  **Stop Loss**: If the spread widens > 10 cents, exit immediately.")
    
    print("\n### EXECUTION LOOP ###")
    print("```python")
    print("def trading_loop(market):")
    print("    spread = market.ask - market.bid")
    print("    volume_ma = market.rolling_volume(20)")
    print("    ")
    print("    # The 'Safe' Entry")
    print("    if spread > 0.05 and market.last_volume > 3 * volume_ma:")
    print("        # Check if 'Yes' + 'No' < 0.98 (True Arb)")
    print("        if market.yes_price + market.no_price < 0.98:")
    print("            execute_arbitrage(market)  # Buy both")
    print("        else:")
    print("            # Directional Scalp based on volume")
    print("            direction = 'Yes' if market.yes_buy_vol > market.no_buy_vol else 'No'")
    print("            place_limit_order(direction, price=market.mid_price)")
    print("```")

try:
    from crewai import Agent, Task, Crew, Process
    # Try importing ChatOpenAI, but handle if it's missing or if API key is missing
    try:
        from langchain_openai import ChatOpenAI
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("No API Key")
    except (ImportError, ValueError):
        # Fallback if no API key or package
        simulate_war_room()
        sys.exit(0)

    # 1. THE AGGRESSIVE QUANT
    quant = Agent(
      role='Aggressive Quant',
      goal='Develop a high-frequency 15m arbitrage strategy for Polymarket.',
      backstory="You are a former HFT trader who thrives on volatility. You believe in 45/45 spreads, fast execution, and exploiting 15-minute crypto price fluctuations on Polymarket. You focus on volume spikes and rapid price convergence.",
      verbose=True,
      allow_delegation=False
    )

    # 2. THE RISK GENERAL
    risk_officer = Agent(
      role='Risk Manager',
      goal='Analyze the Quants strategy for hidden risks like order book depth, fees, and slippage.',
      backstory="You hate losing money. You are extremely skeptical of 'guaranteed' profits. You must point out every reason why the 15m market is dangerous: wide spreads, liquidity gaps, API latency, and resolution disputes.",
      verbose=True,
      allow_delegation=False
    )

    # 3. THE JUDGE
    strategist = Agent(
      role='Chief Strategist',
      goal='Review the debate and output a final, safe execution plan for the bot.',
      backstory="You are the calm voice of reason. You take the Quant's aggressive ideas and the Risk Officer's warnings to create a balanced, profitable, and safe trading algorithm. You care about sustainable alpha.",
      verbose=True,
      allow_delegation=True
    )

    # THE TASKS
    task1 = Task(
        description="""
        Propose a concrete 15-minute crypto arbitrage strategy for Polymarket. 
        Focus on:
        1. Identifying 'Yes' and 'No' shares that sum to less than $1.00 (arbitrage).
        2. Exploiting volume spikes (3x average) to predict short-term direction.
        3. Using a tight 5-cent spread logic for entry.
        4. Provide pseudo-code or specific logic for the entry trigger.
        """,
        agent=quant,
        expected_output="A detailed aggressive strategy proposal with entry triggers and expected profit."
    )

    task2 = Task(
        description="""
        Critique the Quant's strategy. Find 3 specific ways this strategy fails in real Polymarket crypto markets.
        Consider:
        1. Transaction fees (taker vs maker) and how they eat the 5-cent spread.
        2. Liquidity risk: What happens if the order book is thin when you try to exit?
        3. Execution lag: The risk of one leg filling and the other moving (getting 'legged out').
        """,
        agent=risk_officer,
        expected_output="A risk analysis report listing 3 fatal flaws and mitigation warnings."
    )

    task3 = Task(
        description="""
        Synthesize the Quant's strategy and the Risk Officer's critique into a final, safe bot logic.
        1. Define the exact conditions for entry (e.g., minimum profit margin after fees).
        2. Define risk management rules (e.g., max position size, stop loss).
        3. Output the final 'Golden Logic' that balances aggression with safety.
        """,
        agent=strategist,
        expected_output="The final, safe execution plan and logic for the 15-minute Polymarket Arbitrage Bot."
    )

    # START THE WAR ROOM
    war_room = Crew(
      agents=[quant, risk_officer, strategist],
      tasks=[task1, task2, task3],
      verbose=True,
      process=Process.sequential
    )

    print("### STARTING WAR ROOM DEBATE ###")
    result = war_room.kickoff()
    print("\n\n########################")
    print("## FINAL STRATEGY DECISION ##")
    print("########################\n")
    print(result)

except Exception as e:
    # If CrewAI fails to run (e.g., missing LLM), fall back to simulation
    simulate_war_room()
