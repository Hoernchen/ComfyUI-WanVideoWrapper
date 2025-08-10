import { app } from "../../scripts/app.js";

/**
 * Fixed toggle system for WanVideoLoopedGeneration
 * Properly separates widgets from inputs with robust initialization
 */

// Global error handler for input access issues
const originalConsoleError = console.error;
console.error = function(...args) {
    const message = args.join(' ');
    
    // Catch the specific error pattern and provide a helpful hint
    if (message.includes("can't access property \"label\", e.inputs[i] is undefined")) {
        console.warn('WanVideoLoopedGeneration: Detected input access error - this is usually a timing issue during node initialization. Check the reloadNode function.');
    }
    
    // Call original console.error
    originalConsoleError.apply(console, args);
};

// Canvas drawing utilities
function isLowQuality() {
    const canvas = app.canvas;
    return (canvas.ds?.scale || 1) <= 0.5;
}

function drawRoundedRectangle(ctx, options) {
    const lowQuality = isLowQuality();
    ctx.save();
    ctx.strokeStyle = options.colorStroke || LiteGraph.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = options.colorBackground || LiteGraph.WIDGET_BGCOLOR;
    ctx.beginPath();
    ctx.roundRect(
        ...options.pos,
        ...options.size,
        lowQuality ? [0] : options.borderRadius ? [options.borderRadius] : [options.size[1] * 0.5]
    );
    ctx.fill();
    !lowQuality && ctx.stroke();
    ctx.restore();
}

function drawTogglePart(ctx, options) {
    const lowQuality = isLowQuality();
    ctx.save();
    
    const {posX, posY, height, value} = options;
    const toggleRadius = height * 0.36;
    const toggleBgWidth = height * 1.5;
    
    // Toggle Track
    if (!lowQuality) {
        ctx.beginPath();
        ctx.roundRect(posX + 4, posY + 4, toggleBgWidth - 8, height - 8, [height * 0.5]);
        ctx.globalAlpha = app.canvas.editor_alpha * 0.25;
        ctx.fillStyle = "rgba(255,255,255,0.45)";
        ctx.fill();
        ctx.globalAlpha = app.canvas.editor_alpha;
    }
    
    // Toggle itself
    ctx.fillStyle = value === true ? "#89B" : "#888";
    const toggleX = lowQuality || value === false
        ? posX + height * 0.5
        : value === true
            ? posX + height
            : posX + height * 0.75;
    ctx.beginPath();
    ctx.arc(toggleX, posY + height * 0.5, toggleRadius, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.restore();
    return [posX, toggleBgWidth];
}

// Widget class for toggle control - following rgthree pattern
class PromptPairToggleWidget {
    constructor(name, pairNumber) {
        this.name = name;
        // Add safety check for pairNumber to prevent NaN issues
        this.pairNumber = (typeof pairNumber === 'number' && !isNaN(pairNumber)) ? pairNumber : 1;
        this.type = "custom"; // CRITICAL: Must be "custom" like rgthree
        
        // CRITICAL: Value must be an object that gets serialized to Python
        // This is the key insight from rgthree - the entire value object is passed to Python
        this.value = {
            enabled: true,
            pairNumber: this.pairNumber
        };
        
        this.last_y = 0;
        
        // LiteGraph compatibility properties
        this.options = {};
        this.y = 0;
        
        // Hit area for mouse interaction
        this._toggleBounds = null;
    }
    
    draw(ctx, node, w, posY, height) {
        this.last_y = posY;
        
        const margin = 10;
        const innerMargin = margin * 0.33;
        const lowQuality = isLowQuality();
        const midY = posY + height * 0.5;
        
        let posX = margin;
        
        // Draw background
        drawRoundedRectangle(ctx, {
            pos: [posX, posY], 
            size: [node.size[0] - margin * 2, height]
        });
        
        // Draw toggle - use value.enabled
        const toggleBounds = drawTogglePart(ctx, {
            posX, posY, height, value: this.value.enabled
        });
        
        // Store bounds for click detection
        this._toggleBounds = [posX, posX + toggleBounds[1]];
        
        posX += toggleBounds[1] + innerMargin;
        
        if (!lowQuality) {
            // Draw label
            if (!this.value.enabled) {
                ctx.globalAlpha = app.canvas.editor_alpha * 0.4;
            }
            
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            
            const label = `Prompt Pair ${this.pairNumber} ${this.value.enabled ? "(Enabled)" : "(Disabled)"}`;
            ctx.fillText(label, posX, midY);
            
            ctx.globalAlpha = app.canvas.editor_alpha;
        }
    }
    
    mouse(event, pos, node) {
        if (event.type === "pointerdown") {
            if (this._toggleBounds && pos[0] >= this._toggleBounds[0] && pos[0] <= this._toggleBounds[1]) {
                this.value.enabled = !this.value.enabled;
                if (node) {
                    node.setDirtyCanvas(true, true);
                }
                return true;
            }
        }
        return false;
    }
    
    computeSize(width) {
        return [width, 25];
    }
    
    // CRITICAL: This is how widget values get passed to Python
    serializeValue(node, index) {
        // Return the full value object - this is what Python receives in kwargs
        return this.value;
    }
    
    computeSize(width) {
        return [width, 25];
    }
}

// Debounce utility
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Robust initialization utilities
function waitForInputsReady(node, maxAttempts = 10, delay = 50) {
    return new Promise((resolve) => {
        let attempts = 0;
        
        const checkInputs = () => {
            attempts++;
            
            // Check if inputs exist and are properly initialized
            if (node.inputs && 
                Array.isArray(node.inputs) && 
                node.inputs.length >= 0) { // Allow empty inputs array for new nodes
                
                // If inputs array exists but is empty, that's ok for new nodes
                if (node.inputs.length === 0) {
                    resolve(true);
                    return;
                }
                
                // If inputs exist, check they're properly formed
                let allInputsValid = true;
                for (let i = 0; i < node.inputs.length; i++) {
                    if (!node.inputs[i] || typeof node.inputs[i] !== 'object' || !node.inputs[i].hasOwnProperty('name')) {
                        allInputsValid = false;
                        break;
                    }
                }
                
                if (allInputsValid) {
                    resolve(true);
                    return;
                }
            }
            
            if (attempts >= maxAttempts) {
                console.warn('WanVideoLoopedGeneration: Inputs not ready after maximum attempts, proceeding anyway');
                // Initialize inputs array if it doesn't exist
                if (!node.inputs) {
                    node.inputs = [];
                }
                resolve(false);
                return;
            }
            
            setTimeout(checkInputs, delay);
        };
        
        checkInputs();
    });
}

function safelyAccessInputs(node, callback) {
    // Defensive input access wrapper
    if (!node.inputs || !Array.isArray(node.inputs)) {
        console.warn('WanVideoLoopedGeneration: Inputs not available for safe access');
        return;
    }
    
    try {
        callback();
    } catch (error) {
        console.error('WanVideoLoopedGeneration: Error during input access:', error);
    }
}

app.registerExtension({
    name: "WanVideoWrapper.WanVideoLoopedGeneration",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WanVideoLoopedGeneration") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            // Override node creation
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.call(this);
                }
                
                // Enable widget serialization
                this.serialize_widgets = true;
                
                // Storage for toggle widgets
                this.toggleWidgets = new Map();
                
                // Initialization state tracking
                this._initializationComplete = false;
                this._pendingStabilization = false;
                
                // Add a safe reloadNode function that handles edge cases
                this.reloadNode = function() {
                    try {
                        // Ensure inputs exist before any reload operations
                        if (!this.inputs) {
                            this.inputs = [];
                        }
                        
                        // Check all inputs are valid objects with labels
                        for (let i = 0; i < this.inputs.length; i++) {
                            if (!this.inputs[i] || typeof this.inputs[i] !== 'object') {
                                console.warn(`WanVideoLoopedGeneration: Invalid input at index ${i}, creating safe placeholder`);
                                this.inputs[i] = { name: `unknown_${i}`, type: "STRING", slot: i };
                            }
                            
                            // Ensure the input has a label property for compatibility
                            if (!this.inputs[i].hasOwnProperty('label')) {
                                this.inputs[i].label = this.inputs[i].name || `input_${i}`;
                            }
                        }
                        
                        // Mark as ready for operations
                        this._initializationComplete = true;
                        
                        // Run stabilization if needed
                        if (this._pendingStabilization) {
                            this._pendingStabilization = false;
                            this.stabilize();
                        }
                        
                        console.log('WanVideoLoopedGeneration: Node reloaded successfully');
                        
                    } catch (error) {
                        console.error('WanVideoLoopedGeneration: Error in reloadNode:', error);
                        // Ensure we don't get stuck in an error state
                        this._initializationComplete = true;
                    }
                };
                
                // Override configure to restore widget states - following rgthree pattern
                const originalConfigure = this.configure;
                this.configure = function(info) {
                    // Set flag to prevent initial setup conflicts
                    this.configuring = true;
                    this._initializationComplete = false;
                    
                    // CRITICAL: Store Python-defined widgets before clearing
                    const pythonWidgets = [];
                    if (this.widgets && Array.isArray(this.widgets)) {
                        // Save widgets that were created from Python INPUT_TYPES
                        for (const widget of this.widgets) {
                            if (widget && widget.type !== "custom" && 
                                (widget.name === "model_name" || 
                                 widget.name === "precision" || 
                                 widget.name === "quantization" || 
                                 widget.name === "use_disk_cache" || 
                                 widget.name === "device")) {
                                pythonWidgets.push(widget);
                            }
                        }
                    }
                    
                    // Clear only our custom toggle widgets, not Python-defined widgets
                    if (this.widgets && Array.isArray(this.widgets)) {
                        for (let i = this.widgets.length - 1; i >= 0; i--) {
                            const widget = this.widgets[i];
                            if (widget && widget.type === "custom" && widget.name && widget.name.startsWith("toggle_")) {
                                this.widgets.splice(i, 1);
                            }
                        }
                    }
                    this.toggleWidgets.clear();
                    
                    // CRITICAL: Clear only dynamic prompt inputs, not Python-defined inputs
                    // This prevents duplicates when configure restores saved inputs
                    if (this.inputs && Array.isArray(this.inputs)) {
                        for (let i = this.inputs.length - 1; i >= 0; i--) {
                            const input = this.inputs[i];
                            // Only remove dynamic prompt inputs, keep everything else
                            if (input && input.name && input.name.match(/^(positive|negative)_\d{2}$/)) {
                                this.removeInput(i);
                            }
                        }
                    }
                    
                    // Call original configure to set up basic node structure
                    if (originalConfigure) {
                        originalConfigure.call(this, info);
                    }
                    
                    // CRITICAL: Ensure Python widgets are still present after configure
                    // ComfyUI's configure might have recreated the widgets array
                    if (pythonWidgets.length > 0) {
                        // Check if Python widgets are missing and need to be restored
                        const currentPythonWidgets = new Set();
                        if (this.widgets && Array.isArray(this.widgets)) {
                            for (const widget of this.widgets) {
                                if (widget && widget.type !== "custom" && pythonWidgets.some(pw => pw.name === widget.name)) {
                                    currentPythonWidgets.add(widget.name);
                                }
                            }
                        }
                        
                        // Log if any Python widgets are missing
                        for (const pythonWidget of pythonWidgets) {
                            if (!currentPythonWidgets.has(pythonWidget.name)) {
                                console.warn(`WanVideoLoopedGeneration: Python widget '${pythonWidget.name}' was removed during configure!`);
                            }
                        }
                    }
                    
                    // Wait for inputs to be ready before proceeding with widget setup
                    waitForInputsReady(this).then((inputsReady) => {
                        if (!inputsReady) {
                            console.warn('WanVideoLoopedGeneration: Proceeding with initialization despite inputs not being fully ready');
                        }
                        
                        // Restore toggle widgets from serialized data
                        // This happens AFTER inputs are created by the original configure
                        if (info.widgets_values && Array.isArray(info.widgets_values)) {
                            for (const value of info.widgets_values) {
                                // Only process toggle widget values
                                if (typeof value === 'object' && value !== null && 
                                    value.hasOwnProperty('enabled') && value.hasOwnProperty('pairNumber')) {
                                    
                                    const pairNumber = value.pairNumber;
                                    if (pairNumber && pairNumber >= 1 && pairNumber <= 20) {
                                        // Create the toggle widget
                                        const toggleWidget = new PromptPairToggleWidget(`toggle_${pairNumber}`, pairNumber);
                                        
                                        // Restore its value
                                        toggleWidget.value = { ...value };
                                        
                                        // Add to node
                                        if (!this.widgets) {
                                            this.widgets = [];
                                        }
                                        this.widgets.push(toggleWidget);
                                        this.toggleWidgets.set(pairNumber, toggleWidget);
                                    }
                                }
                            }
                        }
                        
                        // Ensure we have at least the first toggle widget
                        if (this.toggleWidgets.size === 0) {
                            // Check if toggle_1 already exists in widgets array
                            const existingToggle1 = this.widgets && this.widgets.find(w => 
                                w && w.type === "custom" && w.name === "toggle_1"
                            );
                            
                            if (!existingToggle1) {
                                const firstToggle = new PromptPairToggleWidget("toggle_1", 1);
                                if (!this.widgets) {
                                    this.widgets = [];
                                }
                                this.widgets.push(firstToggle);
                                this.toggleWidgets.set(1, firstToggle);
                            } else {
                                // Restore existing widget to map
                                this.toggleWidgets.set(1, existingToggle1);
                            }
                        }
                        
                        // CRITICAL: Deduplicate inputs after configure
                        // This handles the case where configure might have restored duplicates
                        this.deduplicateInputs();
                        
                        // CRITICAL: Deduplicate widgets as well
                        this.deduplicateWidgets();
                        
                        // Clear configuration flag and mark initialization complete
                        this.configuring = false;
                        this._initializationComplete = true;
                        
                        // If there was a pending stabilization, run it now
                        if (this._pendingStabilization) {
                            this._pendingStabilization = false;
                            this.stabilize();
                        }
                    });
                };
                
                // Deduplication method - removes duplicate inputs while preserving connections
                this.deduplicateInputs = function() {
                    if (!this.inputs || !Array.isArray(this.inputs)) return;
                    
                    const seen = new Map(); // name -> first occurrence
                    const toRemove = []; // indices to remove
                    
                    // Find duplicates
                    for (let i = 0; i < this.inputs.length; i++) {
                        const input = this.inputs[i];
                        if (!input || !input.name) continue;
                        
                        if (seen.has(input.name)) {
                            // This is a duplicate
                            const firstIdx = seen.get(input.name);
                            const firstInput = this.inputs[firstIdx];
                            
                            // Transfer connection from duplicate to first occurrence if needed
                            if (input.link && !firstInput.link) {
                                firstInput.link = input.link;
                            }
                            
                            toRemove.push(i);
                            console.warn(`WanVideoLoopedGeneration: Removing duplicate input ${input.name} at index ${i}`);
                        } else {
                            seen.set(input.name, i);
                        }
                    }
                    
                    // Remove duplicates in reverse order to preserve indices
                    for (let i = toRemove.length - 1; i >= 0; i--) {
                        this.removeInput(toRemove[i]);
                    }
                    
                    if (toRemove.length > 0) {
                        console.log(`WanVideoLoopedGeneration: Removed ${toRemove.length} duplicate inputs`);
                        this.sortInputs();
                    }
                };
                
                // Deduplication method for widgets
                this.deduplicateWidgets = function() {
                    if (!this.widgets || !Array.isArray(this.widgets)) return;
                    
                    const seen = new Map(); // name -> first occurrence widget
                    const toRemove = []; // indices to remove
                    
                    // Find duplicate widgets - but ONLY our custom toggle widgets
                    for (let i = 0; i < this.widgets.length; i++) {
                        const widget = this.widgets[i];
                        if (!widget || !widget.name) continue;
                        
                        // CRITICAL: Only deduplicate our custom toggle widgets
                        // Never touch Python-defined widgets
                        if (widget.type === "custom" && widget.name.startsWith("toggle_")) {
                            if (seen.has(widget.name)) {
                                // This is a duplicate
                                toRemove.push(i);
                                console.warn(`WanVideoLoopedGeneration: Found duplicate widget ${widget.name} at index ${i}`);
                            } else {
                                seen.set(widget.name, widget);
                                // Ensure widget is in the toggleWidgets map
                                if (widget.pairNumber && !this.toggleWidgets.has(widget.pairNumber)) {
                                    this.toggleWidgets.set(widget.pairNumber, widget);
                                }
                            }
                        }
                    }
                    
                    // Remove duplicates in reverse order to preserve indices
                    for (let i = toRemove.length - 1; i >= 0; i--) {
                        this.widgets.splice(toRemove[i], 1);
                    }
                    
                    if (toRemove.length > 0) {
                        console.log(`WanVideoLoopedGeneration: Removed ${toRemove.length} duplicate toggle widgets`);
                        this.setDirtyCanvas(true);
                    }
                };
                
                // Stabilize method - manages input pairs
                this.stabilize = function() {
                    // CRITICAL: Multiple checks for graph availability
                    if (!this.graph) {
                        console.warn('WanVideoLoopedGeneration: Stabilize called when node.graph is null, skipping');
                        return;
                    }
                    
                    // Check if initialization is complete to avoid race conditions
                    if (!this._initializationComplete) {
                        this._pendingStabilization = true;
                        return;
                    }
                    
                    // Use safe input access wrapper
                    safelyAccessInputs(this, () => {
                        // Ensure inputs array exists and is properly initialized
                        if (!this.inputs) {
                            this.inputs = [];
                        }
                        if (!Array.isArray(this.inputs)) {
                            this.inputs = [];
                        }
                        
                        // Find all prompt pairs
                        const pairs = new Map();
                        let highestConnectedPair = 0;
                        let hasAnyConnection = false;
                        
                        // Safely iterate through inputs with enhanced null checks
                        for (let i = 0; i < this.inputs.length; i++) {
                            const input = this.inputs[i];
                            if (!input || typeof input !== 'object' || !input.name || typeof input.name !== 'string') {
                                continue;
                            }
                            
                            const match = input.name.match(/^(positive|negative)_(\d{2})$/);
                            if (match) {
                                const pairNum = parseInt(match[2], 10);
                                
                                // Add NaN check to prevent arithmetic errors
                                if (isNaN(pairNum) || pairNum < 1 || pairNum > 20) {
                                    console.warn('WanVideoLoopedGeneration: Invalid pair number detected:', match[2]);
                                    continue;
                                }
                                
                                if (!pairs.has(pairNum)) {
                                    pairs.set(pairNum, {hasConnection: false});
                                }
                                
                                if (input.link !== null && input.link !== undefined) {
                                    pairs.get(pairNum).hasConnection = true;
                                    // Safety check before Math.max to prevent NaN propagation
                                    if (!isNaN(pairNum)) {
                                        highestConnectedPair = Math.max(highestConnectedPair, pairNum);
                                    }
                                    hasAnyConnection = true;
                                }
                            }
                        }
                        
                        // Remove unconnected pairs from the end
                        const sortedPairs = Array.from(pairs.entries()).sort((a, b) => b[0] - a[0]);
                        for (const [pairNum, pair] of sortedPairs) {
                            if (pairNum > 1 && !pair.hasConnection && pairNum > highestConnectedPair) {
                                // Remove inputs with enhanced null checks
                                const pairStr = pairNum.toString().padStart(2, '0');
                                const posInput = this.inputs.find(i => i && typeof i === 'object' && i.name === `positive_${pairStr}`);
                                const negInput = this.inputs.find(i => i && typeof i === 'object' && i.name === `negative_${pairStr}`);
                                
                                if (posInput) {
                                    const idx = this.inputs.indexOf(posInput);
                                    if (idx >= 0) this.removeInput(idx);
                                }
                                if (negInput) {
                                    const idx = this.inputs.indexOf(negInput);
                                    if (idx >= 0) this.removeInput(idx);
                                }
                                
                                // Remove toggle widget
                                if (this.widgets && Array.isArray(this.widgets)) {
                                    const toggleWidget = this.widgets.find(w => w && w.name === `toggle_${pairNum}`);
                                    if (toggleWidget) {
                                        const idx = this.widgets.indexOf(toggleWidget);
                                        if (idx >= 0) this.widgets.splice(idx, 1);
                                    }
                                }
                                
                                this.toggleWidgets.delete(pairNum);
                            }
                        }
                        
                        // Determine target number of pairs with safety checks
                        let targetPairs = 1; // Default to at least 1 pair
                        if (hasAnyConnection && !isNaN(highestConnectedPair) && highestConnectedPair >= 0) {
                            targetPairs = highestConnectedPair + 1;
                        }
                        
                        // Ensure targetPairs is within valid bounds
                        targetPairs = Math.max(1, Math.min(targetPairs, 20));
                        
                        // Add missing pairs - ENSURE PAIRS ARE ADDED TOGETHER IN CORRECT ORDER
                        // Process pairs in order and add both positive and negative before moving to next pair
                        for (let pairNum = 1; pairNum <= targetPairs && pairNum <= 20; pairNum++) {
                            const pairStr = pairNum.toString().padStart(2, '0');
                            
                            // Check what's missing for this pair - use strict equality check to prevent duplicates
                            const positives = this.inputs.filter(i => i && typeof i === 'object' && i.name === `positive_${pairStr}`);
                            const negatives = this.inputs.filter(i => i && typeof i === 'object' && i.name === `negative_${pairStr}`);
                            
                            // Log if we find duplicates
                            if (positives.length > 1) {
                                console.warn(`WanVideoLoopedGeneration: Found ${positives.length} duplicate positive_${pairStr} inputs!`);
                            }
                            if (negatives.length > 1) {
                                console.warn(`WanVideoLoopedGeneration: Found ${negatives.length} duplicate negative_${pairStr} inputs!`);
                            }
                            
                            // CRITICAL: Add both positive and negative for this pair before moving to next pair
                            // This prevents interleaving of pairs during rapid stabilization
                            if (positives.length === 0) {
                                this.addInput(`positive_${pairStr}`, "STRING");
                            }
                            if (negatives.length === 0) {
                                this.addInput(`negative_${pairStr}`, "STRING");
                            }
                            
                            // Add toggle widget if missing
                            if (!this.toggleWidgets.has(pairNum)) {
                                // CRITICAL: Also check if widget already exists in widgets array
                                // This prevents duplicate widgets when stabilize is called multiple times
                                const existingWidget = this.widgets && this.widgets.find(w => 
                                    w && w.type === "custom" && w.name === `toggle_${pairNum}`
                                );
                                
                                if (existingWidget) {
                                    // Widget exists in array but not in map - restore it to map
                                    this.toggleWidgets.set(pairNum, existingWidget);
                                } else {
                                    // Check if this pair has any connections
                                    const hasConnection = this.inputs.some(input => 
                                        input && 
                                        typeof input === 'object' &&
                                        (input.name === `positive_${pairStr}` || input.name === `negative_${pairStr}`) && 
                                        input.link !== null && input.link !== undefined
                                    );
                                    
                                    // Only create toggle for pair 1 or pairs with connections
                                    if (pairNum === 1 || hasConnection) {
                                        const toggleWidget = new PromptPairToggleWidget(`toggle_${pairNum}`, pairNum);
                                        
                                        // Properly add widget to node
                                        if (typeof this.addCustomWidget === 'function') {
                                            this.addCustomWidget(toggleWidget);
                                        } else {
                                            // Fallback: add directly to widgets array
                                            if (!this.widgets) {
                                                this.widgets = [];
                                            }
                                            this.widgets.push(toggleWidget);
                                        }
                                        
                                        this.toggleWidgets.set(pairNum, toggleWidget);
                                    }
                                }
                            }
                        }
                        
                        // Remove toggle widgets for pairs that no longer have connections
                        const togglesToRemove = [];
                        for (const [pairNum, toggleWidget] of this.toggleWidgets) {
                            const pairStr = pairNum.toString().padStart(2, '0');
                            const hasConnection = this.inputs.some(input => 
                                input && 
                                typeof input === 'object' &&
                                (input.name === `positive_${pairStr}` || input.name === `negative_${pairStr}`) && 
                                input.link !== null && input.link !== undefined
                            );
                            
                            if (!hasConnection) {
                                togglesToRemove.push(pairNum);
                            }
                        }
                        
                        // Remove toggles for disconnected pairs
                        for (const pairNum of togglesToRemove) {
                            const toggleWidget = this.toggleWidgets.get(pairNum);
                            if (toggleWidget && this.widgets && Array.isArray(this.widgets)) {
                                const idx = this.widgets.indexOf(toggleWidget);
                                if (idx >= 0) {
                                    this.widgets.splice(idx, 1);
                                }
                            }
                            this.toggleWidgets.delete(pairNum);
                        }
                        
                        this.sortInputs();
                        
                        // Final widget deduplication to catch any race conditions
                        this.deduplicateWidgets();
                    }); // End of safelyAccessInputs wrapper
                };
                
                // Sort inputs properly
                this.sortInputs = function() {
                    safelyAccessInputs(this, () => {
                        if (!this.inputs || !Array.isArray(this.inputs)) return;
                        
                        // CRITICAL: Do NOT reorder Python-defined inputs
                        // Only sort the dynamic prompt inputs among themselves
                        
                        // Find the insertion point - after the last Python-defined input
                        let insertionIndex = 0;
                        for (let i = 0; i < this.inputs.length; i++) {
                            const input = this.inputs[i];
                            if (input && input.name && !input.name.match(/^(positive|negative)_\d{2}$/)) {
                                insertionIndex = i + 1;
                            }
                        }
                        
                        // Extract only prompt inputs for sorting
                        const promptInputs = [];
                        const indicesToRemove = [];
                        
                        for (let i = 0; i < this.inputs.length; i++) {
                            const input = this.inputs[i];
                            if (input && typeof input === 'object' && input.name && 
                                typeof input.name === 'string' && input.name.match(/^(positive|negative)_\d{2}$/)) {
                                promptInputs.push(input);
                                indicesToRemove.push(i);
                            }
                        }
                        
                        // Sort prompt inputs by pair number and type - POSITIVE BEFORE NEGATIVE
                        promptInputs.sort((a, b) => {
                            const matchA = a.name.match(/^(positive|negative)_(\d{2})$/);
                            const matchB = b.name.match(/^(positive|negative)_(\d{2})$/);
                            
                            if (!matchA || !matchB) return 0;
                            
                            const pairA = parseInt(matchA[2], 10);
                            const pairB = parseInt(matchB[2], 10);
                            
                            // Add NaN checks to prevent sorting errors
                            if (isNaN(pairA) || isNaN(pairB)) {
                                console.warn('WanVideoLoopedGeneration: Invalid pair numbers in sort:', matchA[2], matchB[2]);
                                return 0;
                            }
                            
                            // First sort by pair number
                            if (pairA !== pairB) return pairA - pairB;
                            
                            // Within the same pair, positive comes first (-1), negative comes second (1)
                            if (matchA[1] === 'positive' && matchB[1] === 'negative') return -1;
                            if (matchA[1] === 'negative' && matchB[1] === 'positive') return 1;
                            
                            return 0;
                        });
                        
                        // Remove prompt inputs from their current positions (in reverse order to preserve indices)
                        for (let i = indicesToRemove.length - 1; i >= 0; i--) {
                            this.inputs.splice(indicesToRemove[i], 1);
                        }
                        
                        // Insert sorted prompt inputs at the correct position
                        for (let i = 0; i < promptInputs.length; i++) {
                            this.inputs.splice(insertionIndex + i, 0, promptInputs[i]);
                        }
                        
                        // Update slot indices with enhanced safety
                        for (let i = 0; i < this.inputs.length; i++) {
                            if (this.inputs[i] && typeof this.inputs[i] === 'object') {
                                this.inputs[i].slot = i;
                            }
                        }
                    });
                };
                
                // Override execution to pass toggle states
                const originalOnExecuted = this.onExecuted;
                this.onExecuted = function(data) {
                    if (originalOnExecuted) {
                        originalOnExecuted.call(this, data);
                    }
                };
                
                // Override getExtraMenuOptions to handle toggle states
                const originalGetExtraMenuOptions = this.getExtraMenuOptions;
                this.getExtraMenuOptions = function(canvas, options) {
                    const menuOptions = originalGetExtraMenuOptions ? 
                        originalGetExtraMenuOptions.call(this, canvas, options) : [];
                    
                    // Add option to enable/disable all
                    let allEnabled = true;
                    let hasAny = false;
                    
                    for (const [pairNum, widget] of this.toggleWidgets) {
                        hasAny = true;
                        if (!widget.value.enabled) {
                            allEnabled = false;
                            break;
                        }
                    }
                    
                    if (hasAny) {
                        menuOptions.push(null); // Separator
                        menuOptions.push({
                            content: allEnabled ? "Disable All Prompts" : "Enable All Prompts",
                            callback: () => {
                                const newState = !allEnabled;
                                for (const [pairNum, widget] of this.toggleWidgets) {
                                    widget.value.enabled = newState;
                                }
                                this.setDirtyCanvas(true);
                            }
                        });
                    }
                    
                    return menuOptions;
                };
                
                // Stabilization on connection change - reduce debounce to minimize race conditions
                this.stabilizeBound = debounce(() => {
                    // Additional safety check in debounced function
                    if (!this.graph) {
                        console.warn('WanVideoLoopedGeneration: Debounced stabilize called when node.graph is null, skipping');
                        return;
                    }
                    this.stabilize();
                }, 50);
                
                const originalOnConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function(type, slotIndex, isConnected, linkInfo, ioSlot) {
                    // CRITICAL: Add null checks before any graph operations
                    if (!this.graph) {
                        console.warn('WanVideoLoopedGeneration: onConnectionsChange called when node.graph is null, skipping');
                        return;
                    }
                    
                    // Add safety checks for numeric values to prevent NaN
                    if (typeof type !== 'number' || isNaN(type)) {
                        console.warn('WanVideoLoopedGeneration: Invalid type parameter in onConnectionsChange:', type);
                        return;
                    }
                    
                    if (typeof slotIndex !== 'number' || isNaN(slotIndex)) {
                        console.warn('WanVideoLoopedGeneration: Invalid slotIndex parameter in onConnectionsChange:', slotIndex);
                        return;
                    }
                    
                    if (originalOnConnectionsChange) {
                        originalOnConnectionsChange.call(this, type, slotIndex, isConnected, linkInfo, ioSlot);
                    }
                    
                    // Only proceed with stabilization if we have a valid graph and valid input type
                    if (type === LiteGraph.INPUT && this.graph) {
                        this.stabilizeBound();
                    }
                };
                
                // Initialize the node safely, waiting for proper setup
                const initializeNode = async () => {
                    // Wait for the graph to be ready
                    await new Promise(resolve => {
                        if (this.graph) {
                            resolve();
                        } else {
                            const checkGraph = () => {
                                if (this.graph) {
                                    resolve();
                                } else {
                                    setTimeout(checkGraph, 10);
                                }
                            };
                            checkGraph();
                        }
                    });
                    
                    // Only add initial pair if not being configured from saved data
                    if (!this.configuring) {
                        // Wait for inputs to be ready before adding initial setup
                        await waitForInputsReady(this, 5, 20); // Shorter wait for new nodes
                        
                        // Check if inputs already exist (from Python definition)
                        const hasPositive01 = this.inputs && this.inputs.some(i => i && i.name === "positive_01");
                        const hasNegative01 = this.inputs && this.inputs.some(i => i && i.name === "negative_01");
                        
                        // Add initial pair only if they don't exist - POSITIVE FIRST, THEN NEGATIVE
                        if (!hasPositive01) {
                            this.addInput("positive_01", "STRING");
                        }
                        if (!hasNegative01) {
                            this.addInput("negative_01", "STRING");
                        }
                        
                        // Sort inputs immediately after adding initial pair
                        this.sortInputs();
                        
                        // Mark initialization complete before stabilization
                        this._initializationComplete = true;
                        
                        // Run initial stabilization - this will create the toggle widget
                        this.stabilize();
                    }
                };
                
                // Start initialization process
                initializeNode().catch(error => {
                    console.error('WanVideoLoopedGeneration: Error during initialization:', error);
                    // Mark as complete even if failed to prevent hanging
                    this._initializationComplete = true;
                });
            };
            
            // No need to override onExecute - widget values are automatically serialized
            // and passed to Python as kwargs when serialize_widgets is true
        }
    }
});