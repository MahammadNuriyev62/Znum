import os
import sys
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import numpy as np
import uvicorn
import json

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "znum", "znum"))
)
from znum.Promethee import Promethee
from helper.Beast import Beast


# Create the FastAPI app
app = FastAPI(title="Excel Data Processor")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")


# Route for the home page
@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route for uploading Excel files
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()

    # Create a pandas DataFrame from the Excel file WITHOUT using the first row as headers
    df = pd.read_excel(io.BytesIO(contents), header=None)

    # Handle NaN values by replacing them with None (will be converted to null in JSON)
    df = df.replace({np.nan: None})

    # Convert the DataFrame to a list of lists (array of arrays)
    data = df.values.tolist()

    # Generate generic column headers (just for internal use)
    headers = [f"Column_{i}" for i in range(len(data[0]))] if data else []

    return {"headers": headers, "data": data, "success": True}


# Route for algorithm execution
@app.post("/execute")
async def execute_algorithm(algorithm: str = Form(...), data: str = Form(...)):
    # Parse the JSON string to get the data
    parsed_data = json.loads(data)

    # Placeholder for algorithm implementation
    if algorithm.lower() == "promethee":
        table = Beast.parse_znums_from_table(parsed_data)
        promethee = Promethee(table, shouldNormalizeWeight=True)
        promethee.solve()
        indices = promethee.ordered_indices

    else:  # TOPSIS
        return JSONResponse(
            content={"message": "Хусюн пока не умеет это делать!", "success": False},
            status_code=501,
        )

    alternatives = parsed_data[1:-1]
    sorted_data = (
        [parsed_data[0]] + [alternatives[i] for i in indices] + [parsed_data[-1]]
    )

    return {
        "original_data": parsed_data,
        "sorted_data": sorted_data,
        "indices": indices,
        "success": True,
    }


# Route for exporting results
@app.post("/export")
async def export_data(format: str = Form(...), data: str = Form(...)):
    # Parse the JSON string to get the data
    parsed_data = json.loads(data)

    # Create a pandas DataFrame from the data without specific column names
    df = pd.DataFrame(parsed_data)

    # Create a buffer to store the file
    buffer = io.BytesIO()

    # Export to the requested format without column headers (as they're already in the data)
    if format.lower() == "xlsx":
        df.to_excel(buffer, index=False, header=False)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = "results.xlsx"
    else:  # CSV
        df.to_csv(buffer, index=False, header=False)
        media_type = "text/csv"
        filename = "results.csv"

    # Reset buffer position to the beginning
    buffer.seek(0)

    # Return a streaming response with the file
    return StreamingResponse(
        buffer,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
