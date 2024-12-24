import pandas as pd
from langchain_core.tools import tool
from typing import Union, Optional
import sqlite3
from langchain_core.runnables.config import ensure_config

db = "travel2.sqlite"


@tool
def search_trip_recommendations(
    location: Optional[str] = None,
    name: Optional[str] = None,
    keywords: Optional[str] = None,
) -> list[dict]:
    """
    Search for trip recommendations based on location, name, and keywords.

    Args:
        location (Optional[str]): The location of the trip recommendation. Defaults to None.
        name (Optional[str]): The name of the trip recommendation. Defaults to None.
        keywords (Optional[str]): The keywords associated with the trip recommendation. Defaults to None.

    Returns:
        list[dict]: A list of trip recommendation dictionaries matching the search criteria.
    """
    config = ensure_config()
    db_session = config["configurable"].get("db_session", None)

    # conn = sqlite3.connect(db)
    # cursor = conn.cursor()

    query = "SELECT * FROM trip_recommendations WHERE 1=1"
    params = []

    if location:
        query += " AND location LIKE ?"
        params.append(f"%{location}%")
    if name:
        query += " AND name LIKE ?"
        params.append(f"%{name}%")
    if keywords:
        keyword_list = keywords.split(",")
        keyword_conditions = " OR ".join(["keywords LIKE ?" for _ in keyword_list])
        query += f" AND ({keyword_conditions})"
        params.extend([f"%{keyword.strip()}%" for keyword in keyword_list])

    results = pd.read_sql(sql=query, con=db_session)
    # cursor.execute(query, params)
    # results = cursor.fetchall()
    #
    # conn.close()
    return results.to_dict(orient="records")
    # return [
    #     dict(zip([column[0] for column in cursor.description], row)) for row in results
    # ]


@tool
def book_excursion(recommendation_id: int) -> str:
    """
    Book a excursion by its recommendation ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to book.

    Returns:
        str: A message indicating whether the trip recommendation was successfully booked or not.
    """
    config = ensure_config()
    db_session = config["configurable"].get("db_session", None)
    # conn = sqlite3.connect(db)
    # cursor = conn.cursor()

    # cursor.execute(
    #     "UPDATE trip_recommendations SET booked = 1 WHERE id = ?", (recommendation_id,)
    # )
    pd.read_sql(sql="UPDATE trip_recommendations SET booked = 1 WHERE id = '{0}'".format(recommendation_id))
    # conn.commit()

    # if cursor.rowcount > 0:
    #     conn.close()
    return f"Trip recommendation {recommendation_id} successfully booked."
    # else:
    #     conn.close()
    #     return f"No trip recommendation found with ID {recommendation_id}."


@tool
def update_excursion(recommendation_id: int, details: str) -> str:
    """
    Update a trip recommendation's details by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to update.
        details (str): The new details of the trip recommendation.

    Returns:
        str: A message indicating whether the trip recommendation was successfully updated or not.
    """
    config = ensure_config()
    db_session = config["configurable"].get("db_session", None)
    # conn = sqlite3.connect(db)
    # cursor = conn.cursor()

    # cursor.execute(
    #     "UPDATE trip_recommendations SET details = ? WHERE id = ?",
    #     (details, recommendation_id),
    # )
    pd.read_sql(sql="UPDATE trip_recommendations SET details = '{0}' WHERE id = '{1}'".
                format(details, recommendation_id))
    # conn.commit()

    # if cursor.rowcount > 0:
        # conn.close()
    return f"Trip recommendation {recommendation_id} successfully updated."
    # else:
    #     # conn.close()
    #     return f"No trip recommendation found with ID {recommendation_id}."


@tool
def cancel_excursion(recommendation_id: int) -> str:
    """
    Cancel a trip recommendation by its ID.

    Args:
        recommendation_id (int): The ID of the trip recommendation to cancel.

    Returns:
        str: A message indicating whether the trip recommendation was successfully cancelled or not.
    """
    config = ensure_config()
    db_session = config["configurable"].get("db_session", None)
    # conn = sqlite3.connect(db)
    # cursor = conn.cursor()

    # cursor.execute(
    #     "UPDATE trip_recommendations SET booked = 0 WHERE id = ?", (recommendation_id,)
    # )
    pd.read_sql(sql="UPDATE trip_recommendations SET booked = 0 WHERE id = '{0}'".format(recommendation_id),
                con=db_session)
    # conn.commit()

    # if cursor.rowcount > 0:
    #     conn.close()
    return f"Trip recommendation {recommendation_id} successfully cancelled."
    # else:
    #     conn.close()
    #     return f"No trip recommendation found with ID {recommendation_id}."



