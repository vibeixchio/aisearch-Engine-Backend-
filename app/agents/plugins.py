from pydantic import BaseModel
from typing import Dict, Any, Callable

class ActionPlugin(BaseModel):
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    
class PluginRegistry:
    def __init__(self):
        self.plugins: Dict[str, ActionPlugin] = {}
        self.callbacks: Dict[str, Callable] = {}
        
    def register(self, plugin: ActionPlugin, callback: Callable):
        self.plugins[plugin.name] = plugin
        self.callbacks[plugin.name] = callback
        
    def get_plugin_schemas(self) -> str:
        schemas = []
        for name, p in self.plugins.items():
            schemas.append(f"Plugin: {name}\nDesc: {p.description}\nSchema: {p.parameters_schema}")
        return "\n---\n".join(schemas)
        
    async def execute(self, name: str, params: dict):
        if name not in self.callbacks:
            raise ValueError(f"Plugin {name} not found")
        return await self.callbacks[name](**params)

plugin_registry = PluginRegistry()

# --- Sample Plugins ---

async def booking_mock(destination: str, date: str, max_price: int):
    # In reality, this would call Expedia/Skyscanner API
    return {
        "status": "success", 
        "message": f"Found 3 flights to {destination} on {date} under ${max_price}.",
        "action_required": "confirm_booking",
        "flights": [
            {"airline": "Air Mock", "price": max_price - 50, "time": "10:00 AM"}
        ]
    }

plugin_registry.register(
    ActionPlugin(
        name="travel_booking",
        description="Finds flights based on destination, date, and budget.",
        parameters_schema={
            "destination": "string (Airport Code)",
            "date": "string (YYYY-MM-DD)",
            "max_price": "integer"
        }
    ),
    booking_mock
)
