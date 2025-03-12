document.addEventListener("DOMContentLoaded", function () {
  // DOM Elements
  const uploadContainer = document.getElementById("upload-container");
  const uploadBox = document.getElementById("upload-box");
  const fileInput = document.getElementById("file-input");
  const uploadSection = document.getElementById("upload-section");
  const dataSection = document.getElementById("data-section");
  const originalTable = document.getElementById("original-table");
  const sortedTable = document.getElementById("sorted-table");
  const executeBtn = document.getElementById("execute-btn");
  const restartBtn = document.getElementById("restart-btn");
  const exportXlsxBtn = document.getElementById("export-xlsx");
  const exportCsvBtn = document.getElementById("export-csv");
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tableContainers = document.querySelectorAll(".table-container");
  const algorithmSelect = document.getElementById("algorithm");

  // Global variables
  let uploadedData = null;
  let sortedData = null;

  // Event Listeners for file upload
  uploadBox.addEventListener("dragover", handleDragOver);
  uploadBox.addEventListener("dragleave", handleDragLeave);
  uploadBox.addEventListener("drop", handleDrop);
  fileInput.addEventListener("change", handleFileSelect);
  document.addEventListener("paste", handlePaste);

  // Event Listeners for UI interactions
  executeBtn.addEventListener("click", executeAlgorithm);
  restartBtn.addEventListener("click", restartApp);
  exportXlsxBtn.addEventListener("click", () => exportData("xlsx"));
  exportCsvBtn.addEventListener("click", () => exportData("csv"));

  // Tab navigation
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      // Remove active class from all tabs
      tabBtns.forEach((b) => b.classList.remove("active"));
      tableContainers.forEach((c) => c.classList.remove("active"));

      // Add active class to clicked tab
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    });
  });

  // File Upload Handlers
  function handleDragOver(e) {
    e.preventDefault();
    uploadBox.classList.add("drag-over");
  }

  function handleDragLeave(e) {
    e.preventDefault();
    uploadBox.classList.remove("drag-over");
  }

  function handleDrop(e) {
    e.preventDefault();
    uploadBox.classList.remove("drag-over");

    if (e.dataTransfer.files.length) {
      processFile(e.dataTransfer.files[0]);
    }
  }

  function handleFileSelect(e) {
    if (fileInput.files.length) {
      processFile(fileInput.files[0]);
    }
  }

  function handlePaste(e) {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;

    for (const item of items) {
      if (item.kind === "file" && item.type.includes("spreadsheet")) {
        const file = item.getAsFile();
        processFile(file);
        break;
      }
    }
  }

  // Process the uploaded file
  function processFile(file) {
    if (!file.name.endsWith(".xlsx")) {
      alert("Please upload an Excel (.xlsx) file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    // Show loading state
    uploadBox.innerHTML = '<div class="loading">Processing file...</div>';

    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        if (data.success) {
          uploadedData = data;
          displayData(uploadedData.data, originalTable);
          switchToDataView();
        } else {
          alert("Error processing file");
          resetUploadBox();
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("Error uploading file: " + error.message);
        resetUploadBox();
      });
  }

  // Reset upload box to original state
  function resetUploadBox() {
    uploadBox.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
          <p>Drag and drop your Excel file here</p>
          <p>or</p>
          <label for="file-input" class="upload-button">Choose File</label>
          <input type="file" id="file-input" accept=".xlsx" hidden>
          <p class="small-text">You can also paste from clipboard</p>
      `;

    // Reattach event listener to the new file input
    document
      .getElementById("file-input")
      .addEventListener("change", handleFileSelect);
  }

  // Display data in a table
  function displayData(data, tableElement) {
    // Clear the table
    tableElement.innerHTML = "";

    if (!data || data.length === 0) {
      tableElement.innerHTML = "<tr><td>No data available</td></tr>";
      return;
    }

    // Add data rows (all rows including the first one are treated as data)
    data.forEach((row, rowIndex) => {
      const tr = document.createElement("tr");

      row.forEach((cell, cellIndex) => {
        const td = document.createElement("td");
        // Handle null values, NaN, undefined
        if (
          cell === null ||
          cell === undefined ||
          (typeof cell === "number" && isNaN(cell))
        ) {
          td.textContent = "";
          td.classList.add("empty-cell");
        } else {
          td.textContent = cell;

          // Add styling for numeric cells
          if (typeof cell === "number") {
            td.classList.add("numeric-cell");

            // Format the number if it's a float
            if (cell % 1 !== 0) {
              td.textContent = parseFloat(cell.toFixed(4));
            }
          }
        }
        tr.appendChild(td);
      });

      tableElement.appendChild(tr);
    });
  }

  // Switch from upload view to data view
  function switchToDataView() {
    uploadSection.classList.add("hidden");
    dataSection.classList.remove("hidden");

    // Reset the file input
    fileInput.value = "";
  }

  // Execute the selected algorithm
  function executeAlgorithm() {
    if (!uploadedData || !uploadedData.data || uploadedData.data.length === 0) {
      alert("Please upload data first");
      return;
    }

    const algorithm = algorithmSelect.value;

    // Show loading state
    executeBtn.disabled = true;
    executeBtn.textContent = "Processing...";

    fetch("/execute", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        algorithm: algorithm,
        data: JSON.stringify(uploadedData.data),
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        executeBtn.disabled = false;
        executeBtn.textContent = "Execute";

        if (data.success) {
          sortedData = data.sorted_data;
          displayData(sortedData, sortedTable);

          // Switch to sorted data tab
          tabBtns.forEach((b) => b.classList.remove("active"));
          tableContainers.forEach((c) => c.classList.remove("active"));

          document
            .querySelector('[data-tab="sorted-data"]')
            .classList.add("active");
          document.getElementById("sorted-data").classList.add("active");
        } else {
          alert("Error executing algorithm");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("Error executing algorithm");
        executeBtn.disabled = false;
        executeBtn.textContent = "Execute";
      });
  }

  // Export data
  function exportData(format) {
    if (!sortedData) {
      alert("Please execute algorithm first");
      return;
    }

    const exportBtn = format === "xlsx" ? exportXlsxBtn : exportCsvBtn;
    const originalText = exportBtn.textContent;

    exportBtn.disabled = true;
    exportBtn.textContent = "Exporting...";

    fetch("/export", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: new URLSearchParams({
        format: format,
        data: JSON.stringify(sortedData),
      }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.blob();
      })
      .then((blob) => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.style.display = "none";
        a.href = url;
        a.download = `results.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);

        exportBtn.disabled = false;
        exportBtn.textContent = originalText;
      })
      .catch((error) => {
        console.error("Error:", error);
        alert(`Error exporting ${format}`);
        exportBtn.disabled = false;
        exportBtn.textContent = originalText;
      });
  }

  // Restart the application
  function restartApp() {
    uploadedData = null;
    sortedData = null;
    originalTable.innerHTML = "";
    sortedTable.innerHTML = "";
    fileInput.value = "";

    // Reset upload box
    resetUploadBox();

    // Switch back to upload view
    dataSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");

    // Reset tabs
    tabBtns.forEach((b) => b.classList.remove("active"));
    tableContainers.forEach((c) => c.classList.remove("active"));
    document
      .querySelector('[data-tab="original-data"]')
      .classList.add("active");
    document.getElementById("original-data").classList.add("active");
  }
});
