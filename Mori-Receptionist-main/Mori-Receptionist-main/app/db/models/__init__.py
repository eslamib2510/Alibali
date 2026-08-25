"""Re-export every model so `Base.metadata` knows about all tables.

Alembic's autogenerate scans `Base.metadata`. If a model isn't imported
somewhere by the time autogen runs, its table is missing from the migration.
Importing them all here is the cleanest way to guarantee that.
"""

from app.db.models.conversation import Conversation
from app.db.models.customer import Customer
from app.db.models.knowledge import Knowledge
from app.db.models.message import Message
from app.db.models.tenant import Tenant

__all__ = ["Tenant", "Customer", "Conversation", "Message", "Knowledge"]
