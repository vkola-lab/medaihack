import { Niivue } from 'https://esm.sh/@niivue/niivue';



let Database = {};

// Common Radiopharmaceutical dictionary
const tracerDictionary = {
    "FBP": "Florbetapir (AV-45)",
    "FBB": "Florbetaben",
    "PIB": "Pittsburg Compound-B",
    "NAV": "flutafuranol"
};

async function loadData() {
    try {
        const response = await fetch('./data/train.csv');
        const csvText = await response.text();
        
        const lines = csvText.split('\n');
        const headers = lines[0].split(',');
        
        // Find the exact column indexes dynamically
        const idIndex = headers.findIndex(h => h.trim() === 'ID');
        const tracerIndex = headers.findIndex(h => h.trim() === 'TRACER.AMY');

        // Build the dictionary
        for (let i = 1; i < lines.length; i++) {
            const cols = lines[i].split(',');
            // Ensure the row has data to prevent errors on empty trailing lines
            if (cols.length > Math.max(idIndex, tracerIndex)) { 
                const id = cols[idIndex].trim();
                const tracer = cols[tracerIndex].trim();
                if (id) {
                    Database[id] = tracer;
                }
            }
        }
        console.log("Successfully loaded metadata for", Object.keys(Database).length, "subjects.");
    } catch (error) {
        console.warn("Could not load train.csv. Make sure the file is in your project folder!", error);
    }
}

// Trigger the CSV load as soon as the app starts
loadData();




//------------------------------------------------CSV LOAD ABOVE---------------------------------------------





const sliceOptions = { show3Dcrosshair: true, backColor: [0, 0, 0, 1] };

// 1. Initialize Viewers
const nvCoronal = new Niivue(sliceOptions);
nvCoronal.attachTo('coronal-canvas');
nvCoronal.setSliceType(nvCoronal.sliceTypeCoronal);

const nvSagittal = new Niivue(sliceOptions);
nvSagittal.attachTo('sagittal-canvas');
nvSagittal.setSliceType(nvSagittal.sliceTypeSagittal);

const nvAxial = new Niivue(sliceOptions);
nvAxial.attachTo('axial-canvas');
nvAxial.setSliceType(nvAxial.sliceTypeAxial);

const nvRender = new Niivue({ show3Dcrosshair: true, backColor: [0, 0, 0, 1] });
nvRender.attachTo('render-canvas');
nvRender.setSliceType(nvRender.sliceTypeRender); 

// Sync canvases
const allViews = [nvCoronal, nvSagittal, nvAxial, nvRender];
nvCoronal.syncWith(allViews);
nvSagittal.syncWith(allViews);
nvAxial.syncWith(allViews);
nvRender.syncWith(allViews);

// --- BULLETPROOF HACK: Native Offline .npy Parser ---
function parseNpy(arrayBuffer) {
    const view = new DataView(arrayBuffer);
    const headerLen = view.getUint16(8, true);
    const dataOffset = 10 + headerLen;

    // Read the string header to extract the dynamic shape
    const decoder = new TextDecoder('utf-8');
    const headerStr = decoder.decode(new Uint8Array(arrayBuffer, 10, headerLen));
    
    let shape = [128, 128, 128]; // Fallback
    const shapeMatch = headerStr.match(/'shape':\s*\(([^)]+)\)/);
    if (shapeMatch) {
        shape = shapeMatch[1].split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    }

    const data = new Float32Array(arrayBuffer, dataOffset);
    return { data, shape };
}

// --- BULLETPROOF HACK: In-Memory NIfTI Builder ---
function createNiftiBlob(f32Data, pythonShape) {
    // Reverse Python [Batch, Z, Y, X] to NIfTI [X, Y, Z]
    const dims = pythonShape.slice(-3).reverse();
    const buffer = new ArrayBuffer(352 + f32Data.byteLength);
    const view = new DataView(buffer);
    
    view.setInt32(0, 348, true);         
    view.setInt16(40, 3, true);          
    view.setInt16(42, dims[0], true);    
    view.setInt16(44, dims[1], true);    
    view.setInt16(46, dims[2], true);    
    view.setInt16(48, 1, true);          
    
    view.setInt16(70, 16, true);         // FLOAT32
    view.setInt16(72, 32, true);         // bitpix
    
    // Standardize voxel size to prevent stretching
    view.setFloat32(76, 1.0, true); 
    view.setFloat32(80, 1.0, true); 
    view.setFloat32(84, 1.0, true); 
    view.setFloat32(88, 1.0, true); 
    
    view.setFloat32(108, 352.0, true);   // vox_offset
    
    // Set minimal qform to let Niivue auto-center the volume
    view.setInt16(252, 1, true); 

    // magic bytes 'n+1\0'
    view.setUint8(344, 110); 
    view.setUint8(345, 43);  
    view.setUint8(346, 49);  
    view.setUint8(347, 0);   
    
    // Stitch data precisely using byte offsets
    new Uint8Array(buffer, 352).set(new Uint8Array(f32Data.buffer, f32Data.byteOffset, f32Data.byteLength));
    
    return new Blob([buffer], { type: 'application/octet-stream' });
}


// --- File Upload & Logic ---
document.getElementById('file-input').addEventListener('change', async function(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Parse Subject ID from Filename (e.g., "B98313661.npy" -> "B98313661")
    const fileName = file.name;
    // Splits by underscore or dot to grab just the ID at the front
    const nameParts = fileName.replace('.npy', '').split('_');

    let subjectID = "--";
    
    // Handle the two different cohort naming conventions
    if (nameParts[0] === 'A4') {
        // If it starts with A4_, the ID is the second chunk (e.g., B99987993)
        subjectID = nameParts[1]; 
    } else {
        // Otherwise, assume it's NACC where the ID is the first chunk (e.g., NACC000314)
        subjectID = nameParts[0]; 
    }

    // Look up the tracer in our parsed CSV database
    const tracerAbbrev = Database[subjectID] || "UNKNOWN";
    
    // Map the abbreviation to the full clinical name
    const fullTracerName = tracerDictionary[tracerAbbrev] || tracerAbbrev;

    // Update the UI
    document.getElementById('meta-subject').innerText = subjectID;
    document.getElementById('meta-tracer').innerText = tracerAbbrev;
    document.getElementById('meta-full-tracer').innerText = fullTracerName;

    // Set Loading State
    document.getElementById('prediction-val').innerText = '...';
    const badge = document.getElementById('status-badge');
    badge.innerText = 'CONSTRUCTING 3D TENSOR...';
    badge.className = "mt-4 inline-block px-3 py-1 rounded text-sm font-bold bg-blue-900 text-blue-300 animate-pulse";

    try {
        // 1. Read and Parse the .npy File Natively
        const arrayBuffer = await file.arrayBuffer();
        const npyData = parseNpy(arrayBuffer);

        // 2. Wrap pixels in our NIfTI format
        const niftiBlob = createNiftiBlob(npyData.data, npyData.shape);
        const niftiUrl = URL.createObjectURL(niftiBlob);
        // 3. AUTO-CONTRAST SCANNER
        // Scans the array to find the true min/max
        let min = Infinity, max = -Infinity;
        for(let i=0; i<npyData.data.length; i+=25) {
            const val = npyData.data[i];
            if (val < min) min = val;
            if (val > max) max = val;
        }


        const volumeSettings = [{ 
            url: niftiUrl, 
            colormap: 'actc',
            cal_min: min + ((max - min) )
        }];

        await nvCoronal.loadVolumes(volumeSettings);
        await nvSagittal.loadVolumes(volumeSettings);
        await nvAxial.loadVolumes(volumeSettings);
        await nvRender.loadVolumes(volumeSettings);

        // 4. Simulate AI Inference
        setTimeout(() => {
            const mockScore = (Math.random() * 85) - 5; 
            updateDashboard(mockScore);
        }, 2000);

    } catch (error) {
        console.error("Failed to parse .npy file:", error);
        badge.innerText = "ERROR PARSING TENSOR";
        badge.className = "mt-4 inline-block px-3 py-1 rounded text-sm font-bold bg-red-900 text-red-300";
    }
});

function updateDashboard(score) {
    document.getElementById('prediction-val').innerText = score.toFixed(1);
    const badge = document.getElementById('status-badge');
    const citation = document.getElementById('threshold-cite');

    if (score < 10) {
        badge.innerText = "NO AMYLOID";
        badge.className = "mt-4 inline-block px-3 py-1 rounded text-sm font-bold bg-blue-900 text-blue-300";
        citation.innerHTML = "Excludes pathology with high certainty.<br/><br/><span class='text-gray-400 font-semibold'>(Collij et al. 2024)</span>";
    } else if (score < 30) {
        badge.innerText = "EMERGING";
        badge.className = "mt-4 inline-block px-3 py-1 rounded text-sm font-bold bg-green-900 text-green-300";
        citation.innerHTML = "Optimal for moderate plaque detection.<br/><br/><span class='text-gray-400 font-semibold'>(Dore et al. 2020)</span>";
    } else if (score < 40) {
        badge.innerText = "THERAPY CUTOFF";
        badge.className = "mt-4 inline-block px-3 py-1 rounded text-sm font-bold bg-orange-900 text-orange-300";
        citation.innerHTML = "Consensus range for anti-amyloid therapy.<br/><br/><span class='text-gray-400 font-semibold'>(Alzheimer's Assoc., 2024)</span>";
    } else {
        badge.innerText = "ESTABLISHED";
        badge.className = "mt-4 inline-block px-3 py-1 rounded text-sm font-bold bg-red-900 text-red-300";
        citation.innerHTML = "Corresponds to established visual reads.<br/><br/><span class='text-gray-400 font-semibold'>(Hanseeuw et al. 2021)</span>";
    }
}