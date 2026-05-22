"""MCC (Merchant Category Code) lookup tool"""

import logging
from typing import Any, Dict
from uuid import UUID

from pydantic import BaseModel, Field

from tools.base import BaseTool

logger = logging.getLogger(__name__)


# ============================================================================
# Parameter Schema
# ============================================================================


class MccLookupParams(BaseModel):
    """Parameters for MCC code lookup"""

    mcc_code: str = Field(..., description="4-digit Merchant Category Code to lookup")


# ============================================================================
# MCC Data
# ============================================================================

# Common MCC codes with descriptions and suggested categories.
# Reference: ISO 18245 / Visa & Mastercard MCC tables.
MCC_DATABASE: Dict[str, Dict[str, str]] = {
    # Grocery & Food
    "5411": {"description": "Grocery Stores, Supermarkets", "category": "Groceries"},
    "5422": {"description": "Freezer and Locker Meat Provisioners", "category": "Groceries"},
    "5441": {"description": "Candy, Nut, and Confectionery Stores", "category": "Groceries"},
    "5451": {"description": "Dairy Products Stores", "category": "Groceries"},
    "5462": {"description": "Bakeries", "category": "Groceries"},
    "5499": {"description": "Miscellaneous Food Stores", "category": "Groceries"},
    # Restaurants & Dining
    "5812": {"description": "Eating Places, Restaurants", "category": "Food"},
    "5813": {"description": "Bars, Cocktail Lounges, Taverns", "category": "Food"},
    "5814": {"description": "Fast Food Restaurants", "category": "Food"},
    # Transportation
    "4011": {"description": "Railroads", "category": "Transport"},
    "4111": {"description": "Local and Suburban Commuter Passenger Transportation", "category": "Transport"},
    "4112": {"description": "Passenger Railways", "category": "Transport"},
    "4121": {"description": "Taxicabs and Rideshares", "category": "Transport"},
    "4131": {"description": "Bus Lines", "category": "Transport"},
    "4214": {"description": "Motor Freight Carriers and Trucking", "category": "Transport"},
    "4789": {"description": "Transportation Services (Not Elsewhere Classified)", "category": "Transport"},
    "7512": {"description": "Car Rental Agencies", "category": "Transport"},
    # Gas / Fuel
    "5541": {"description": "Service Stations (with or without Ancillary Services)", "category": "Fuel"},
    "5542": {"description": "Automated Fuel Dispensers", "category": "Fuel"},
    "5983": {"description": "Fuel Dealers (Non-Automotive)", "category": "Fuel"},
    # Airlines & Travel
    "3000": {"description": "Airlines", "category": "Travel"},
    "3001": {"description": "Airlines", "category": "Travel"},
    "4511": {"description": "Airlines, Air Carriers", "category": "Travel"},
    "4722": {"description": "Travel Agencies and Tour Operators", "category": "Travel"},
    "7011": {"description": "Hotels, Motels, and Resorts", "category": "Accommodation"},
    "7012": {"description": "Timeshares", "category": "Accommodation"},
    "7032": {"description": "Sporting and Recreational Camps", "category": "Accommodation"},
    "7033": {"description": "Trailer Parks and Campgrounds", "category": "Accommodation"},
    # Shopping & Retail
    "5200": {"description": "Home Supply Warehouse Stores", "category": "Shopping"},
    "5211": {"description": "Lumber and Building Materials Stores", "category": "Shopping"},
    "5251": {"description": "Hardware Stores", "category": "Shopping"},
    "5261": {"description": "Nurseries and Garden Supply Stores", "category": "Shopping"},
    "5300": {"description": "Wholesale Clubs", "category": "Shopping"},
    "5310": {"description": "Discount Stores", "category": "Shopping"},
    "5311": {"description": "Department Stores", "category": "Shopping"},
    "5331": {"description": "Variety Stores", "category": "Shopping"},
    "5399": {"description": "General Merchandise", "category": "Shopping"},
    "5611": {"description": "Men's and Boys' Clothing Stores", "category": "Shopping"},
    "5621": {"description": "Women's Ready-to-Wear Stores", "category": "Shopping"},
    "5631": {"description": "Women's Accessory and Specialty Stores", "category": "Shopping"},
    "5641": {"description": "Children's and Infants' Wear Stores", "category": "Shopping"},
    "5651": {"description": "Family Clothing Stores", "category": "Shopping"},
    "5661": {"description": "Shoe Stores", "category": "Shopping"},
    "5691": {"description": "Men's and Women's Clothing Stores", "category": "Shopping"},
    "5699": {"description": "Miscellaneous Apparel and Accessory Shops", "category": "Shopping"},
    "5944": {"description": "Jewelry Stores, Watches, Clocks", "category": "Shopping"},
    "5945": {"description": "Hobby, Toy, and Game Shops", "category": "Shopping"},
    # Electronics
    "5732": {"description": "Electronics Stores", "category": "Electronics"},
    "5734": {"description": "Computer Software Stores", "category": "Electronics"},
    "5735": {"description": "Record Stores", "category": "Entertainment"},
    # Health & Pharmacy
    "5912": {"description": "Drug Stores and Pharmacies", "category": "Health"},
    "5975": {"description": "Hearing Aids", "category": "Health"},
    "5976": {"description": "Orthopedic Goods", "category": "Health"},
    "8011": {"description": "Doctors (Not Elsewhere Classified)", "category": "Health"},
    "8021": {"description": "Dentists, Orthodontists", "category": "Health"},
    "8031": {"description": "Osteopathic Physicians", "category": "Health"},
    "8041": {"description": "Chiropractors", "category": "Health"},
    "8042": {"description": "Optometrists, Ophthalmologists", "category": "Health"},
    "8043": {"description": "Opticians, Optical Goods", "category": "Health"},
    "8049": {"description": "Podiatrists, Chiropodists", "category": "Health"},
    "8050": {"description": "Nursing and Personal Care Facilities", "category": "Health"},
    "8062": {"description": "Hospitals", "category": "Health"},
    "8071": {"description": "Medical and Dental Labs", "category": "Health"},
    "8099": {"description": "Medical Services (Not Elsewhere Classified)", "category": "Health"},
    # Education
    "8211": {"description": "Elementary and Secondary Schools", "category": "Education"},
    "8220": {"description": "Colleges, Universities", "category": "Education"},
    "8241": {"description": "Correspondence Schools", "category": "Education"},
    "8244": {"description": "Business and Secretarial Schools", "category": "Education"},
    "8249": {"description": "Vocational and Trade Schools", "category": "Education"},
    "8299": {"description": "Schools and Educational Services (Not Elsewhere Classified)", "category": "Education"},
    # Entertainment
    "7832": {"description": "Motion Picture Theaters", "category": "Entertainment"},
    "7841": {"description": "Video Tape Rental Stores", "category": "Entertainment"},
    "7911": {"description": "Dance Halls, Studios, and Schools", "category": "Entertainment"},
    "7922": {"description": "Theatrical Producers and Ticket Agencies", "category": "Entertainment"},
    "7929": {"description": "Bands, Orchestras, Entertainers", "category": "Entertainment"},
    "7932": {"description": "Billiard and Pool Establishments", "category": "Entertainment"},
    "7933": {"description": "Bowling Alleys", "category": "Entertainment"},
    "7941": {"description": "Commercial Sports, Athletic Fields", "category": "Entertainment"},
    "7991": {"description": "Tourist Attractions and Exhibits", "category": "Entertainment"},
    "7993": {"description": "Video Amusement Game Supplies", "category": "Entertainment"},
    "7994": {"description": "Video Game Arcades", "category": "Entertainment"},
    "7996": {"description": "Amusement Parks, Circuses, Carnivals", "category": "Entertainment"},
    "7997": {"description": "Membership Clubs (Sports, Recreation)", "category": "Entertainment"},
    "7998": {"description": "Aquariums, Seaquariums, Dolphinariums", "category": "Entertainment"},
    "7999": {"description": "Recreation Services (Not Elsewhere Classified)", "category": "Entertainment"},
    # Utilities & Services
    "4812": {"description": "Telecommunication Equipment and Telephone Sales", "category": "Utilities"},
    "4814": {"description": "Telecommunication Services", "category": "Utilities"},
    "4816": {"description": "Computer Network/Information Services", "category": "Utilities"},
    "4899": {"description": "Cable, Satellite, and Other Pay Television", "category": "Utilities"},
    "4900": {"description": "Utilities - Electric, Gas, Water, Sanitary", "category": "Utilities"},
    # Insurance
    "5960": {"description": "Direct Marketing - Insurance Services", "category": "Insurance"},
    "6300": {"description": "Insurance Underwriting, Premiums", "category": "Insurance"},
    # Financial Services
    "6010": {"description": "Financial Institutions - Manual Cash Disbursements", "category": "Finance"},
    "6011": {"description": "Financial Institutions - Automated Cash Disbursements (ATM)", "category": "Finance"},
    "6012": {"description": "Financial Institutions - Merchandise and Services", "category": "Finance"},
    "6051": {"description": "Non-Financial Institutions - Foreign Currency, Money Orders", "category": "Finance"},
    "6211": {"description": "Security Brokers/Dealers", "category": "Finance"},
    # Personal Services
    "7210": {"description": "Laundry, Cleaning, and Garment Services", "category": "Services"},
    "7211": {"description": "Laundry Services - Family and Commercial", "category": "Services"},
    "7216": {"description": "Dry Cleaners", "category": "Services"},
    "7230": {"description": "Barber and Beauty Shops", "category": "Services"},
    "7251": {"description": "Shoe Repair, Hat Cleaning", "category": "Services"},
    "7261": {"description": "Funeral Services, Crematories", "category": "Services"},
    "7277": {"description": "Counseling Services", "category": "Services"},
    "7297": {"description": "Massage Parlors", "category": "Services"},
    "7298": {"description": "Health and Beauty Spas", "category": "Services"},
    # Subscriptions & Digital
    "5815": {"description": "Digital Goods - Media, Books, Movies, Music", "category": "Subscriptions"},
    "5816": {"description": "Digital Goods - Games", "category": "Subscriptions"},
    "5817": {"description": "Digital Goods - Applications", "category": "Subscriptions"},
    "5818": {"description": "Digital Goods - Large Digital Goods Merchant", "category": "Subscriptions"},
    # Government
    "9211": {"description": "Court Costs, Alimony, Child Support", "category": "Government"},
    "9222": {"description": "Fines", "category": "Government"},
    "9223": {"description": "Bail and Bond Payments", "category": "Government"},
    "9311": {"description": "Tax Payments", "category": "Government"},
    "9399": {"description": "Government Services (Not Elsewhere Classified)", "category": "Government"},
    "9402": {"description": "Postal Services - Government Only", "category": "Government"},
}


# ============================================================================
# Tool Implementation
# ============================================================================


class MccLookupTool(BaseTool):
    """Look up information about a Merchant Category Code (MCC)"""

    name = "mcc_lookup"
    description = (
        "Look up information about a Merchant Category Code (MCC). "
        "Returns code, description, and suggested category."
    )
    args_schema = MccLookupParams

    async def execute(self, user_id: UUID, arguments: Dict[str, Any]) -> Dict[str, Any]:
        mcc_code = arguments["mcc_code"].strip()
        entry = MCC_DATABASE.get(mcc_code)

        if entry:
            return {
                "code": mcc_code,
                "description": entry["description"],
                "category": entry["category"],
            }

        return {
            "code": mcc_code,
            "description": "Unknown MCC code",
            "category": "Other",
        }
