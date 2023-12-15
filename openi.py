from datetime import datetime

class Trade:
    def __init__(self, trade_id: int, trade_type: str, quantity: int, price: float):
        self.trade_id = trade_id
        self.trade_type = trade_type
        self.quantity = quantity
        self.price = price
        self.timestamp = datetime.now()
        self.active = True

    def close_trade(self):
        self.active = False


class TradesManager:
    def __init__(self, initial_balance: float = 10000):
        self.balance: float = initial_balance
        self.trades: list[Trade] = []

    def open_trade(self, trade_type: str, quantity: int, price: float):
        price = self.get_price(price)
        cost = quantity * price
        if cost > self.balance:
            return False
        else:
            trade_id = len(self.trades) + 1
            trade = Trade(trade_id, trade_type, quantity, price)
            self.trades.append(trade)
            self.balance -= cost 
            return True

    def close_trade_by_id(self, trade_id: int, current_price: float):
        for trade in self.trades:
            if trade.trade_id == trade_id:
                tradetype = trade.trade_type
                quantity = trade.quantity
                inprice = trade.price
                if tradetype == "long":
                    self.balance += current_price * quantity
                    trade.close_trade()
                    return True
        else:
            print("Trade ID not found!")
            return False

    def get_trade_type_by_id(self, trade_id: int) -> str:
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade.trade_type
        else:
            print("Trade ID not found!")
            return None

    def get_active_trades(self) -> list[Trade]:
        return [trade for trade in self.trades if trade.active]

    def get_price(self, price: float) -> float:
        # Placeholder function to simulate price retrieval
        return price  # You can modify this function for actual price retrieval
