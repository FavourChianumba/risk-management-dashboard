""" Data package for handling data retrieval and processing. """

from .retrieve_data import (
    get_market_data,
    get_fred_data,
    get_yahoo_finance_data,
    get_yahoo_query_data,
    get_alpha_vantage_data,
    load_crisis_periods,
    get_data_for_date_range
)

from .process_data import (
    clean_data,
    calculate_returns,
    create_portfolio,
    align_data
)