const board = document.getElementById('board');
const svgLayer = document.getElementById('connection-layer');
const statusBar = document.getElementById('status-bar');
const selBox = document.getElementById('selection-box');
const contextMenu = document.getElementById('context-menu');

let nodes = {}; 
let connections = []; 
let nodeCounter = 0;
let activeReactionNodeId = null; 

// --- UI 互動：切換折疊選單 ---
window.toggleAccordion = function(contentId, headerElement) {
    const content = document.getElementById(contentId);
    if (content.style.display === 'none') {
        content.style.display = 'flex';
        headerElement.classList.add('active');
    } else {
        content.style.display = 'none';
        headerElement.classList.remove('active');
    }
}

// --- 1. 新增節點 ---
window.createNode = function(chemicalValue, displayName, type, x = null, y = null) {
    nodeCounter++;
    const nodeId = 'node-' + nodeCounter;
    
    if (x === null) x = 50 + (Math.random() * 60);
    if (y === null) y = 50 + (Math.random() * 60);

    const el = document.createElement('div');
    el.className = `node ${type}`;
    el.id = nodeId;
    el.textContent = displayName;
    el.style.left = x + 'px';
    el.style.top = y + 'px';

    el.addEventListener('mousedown', (e) => startDrag(e, nodeId));
    el.addEventListener('click', (e) => handleNodeClick(e, nodeId));
    
    el.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        showContextMenu(e, nodeId);
    });

    board.appendChild(el);
    nodes[nodeId] = { id: nodeId, element: el, value: chemicalValue, x: x, y: y };
    
    statusBar.children[0].textContent = `✨ 已加入 ${displayName}。`;
    return nodeId;
}

// --- 2. 點擊反應邏輯 ---
function handleNodeClick(e, nodeId) {
    if (isDragging) return;
    clearMultiSelection();

    if (!activeReactionNodeId) {
        activeReactionNodeId = nodeId;
        nodes[nodeId].element.classList.add('active-reactant');
        statusBar.children[0].textContent = `🔄 已選擇 [${nodes[nodeId].value}]，請點擊第二個物質...`;
    } else if (activeReactionNodeId === nodeId) {
        nodes[nodeId].element.classList.remove('active-reactant');
        activeReactionNodeId = null;
        statusBar.children[0].textContent = "💡 提示：從左側點擊物質加入，單擊第一個節點，再點擊第二個進行反應。";
    } else {
        const nodeA = nodes[activeReactionNodeId];
        const nodeB = nodes[nodeId];
        
        nodeA.element.classList.remove('active-reactant');
        activeReactionNodeId = null;

        const alreadyConnected = connections.some(c => 
            (c.from === nodeA.id && c.to === nodeB.id) || (c.from === nodeB.id && c.to === nodeA.id)
        );

        if (!alreadyConnected) triggerReaction(nodeA, nodeB);
    }
}

// --- 3. 節點拖曳與連線 ---
let draggedNodeId = null;
let offsetX = 0, offsetY = 0;
let isDragging = false;

function startDrag(e, nodeId) {
    if (e.button !== 0) return; 
    isDragging = false; 
    draggedNodeId = nodeId;
    const el = nodes[nodeId].element;
    offsetX = e.clientX - el.offsetLeft;
    offsetY = e.clientY - el.offsetTop;
    
    document.addEventListener('mousemove', dragNode);
    document.addEventListener('mouseup', stopDragNode);
}

function dragNode(e) {
    if (!draggedNodeId) return;
    isDragging = true;
    let newX = e.clientX - offsetX;
    let newY = e.clientY - offsetY;
    nodes[draggedNodeId].element.style.left = newX + 'px';
    nodes[draggedNodeId].element.style.top = newY + 'px';
    nodes[draggedNodeId].x = newX;
    nodes[draggedNodeId].y = newY;
    drawConnections();
}

function stopDragNode() {
    draggedNodeId = null;
    document.removeEventListener('mousemove', dragNode);
    document.removeEventListener('mouseup', stopDragNode);
    setTimeout(() => isDragging = false, 50); 
}

// --- 4. 拉框多選 ---
let isSelectingBox = false;
let selStartX = 0, selStartY = 0;

board.addEventListener('mousedown', (e) => {
    if (e.button !== 0 || (e.target !== board && e.target !== svgLayer)) return;
    
    isSelectingBox = true;
    clearMultiSelection(); 
    if(activeReactionNodeId) {
        nodes[activeReactionNodeId].element.classList.remove('active-reactant');
        activeReactionNodeId = null;
    }

    const rect = board.getBoundingClientRect();
    selStartX = e.clientX - rect.left;
    selStartY = e.clientY - rect.top;
    
    selBox.style.left = selStartX + 'px';
    selBox.style.top = selStartY + 'px';
    selBox.style.width = '0px';
    selBox.style.height = '0px';
    selBox.style.display = 'block';
});

board.addEventListener('mousemove', (e) => {
    if (!isSelectingBox) return;
    const rect = board.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    const left = Math.min(selStartX, currentX);
    const top = Math.min(selStartY, currentY);
    const width = Math.abs(currentX - selStartX);
    const height = Math.abs(currentY - selStartY);

    selBox.style.left = left + 'px';
    selBox.style.top = top + 'px';
    selBox.style.width = width + 'px';
    selBox.style.height = height + 'px';

    Object.values(nodes).forEach(node => {
        const nx = node.x; const ny = node.y;
        const nw = node.element.offsetWidth; const nh = node.element.offsetHeight;
        
        if (nx < left + width && nx + nw > left && ny < top + height && ny + nh > top) {
            node.element.classList.add('selected');
        } else {
            node.element.classList.remove('selected');
        }
    });
});

board.addEventListener('mouseup', () => {
    isSelectingBox = false;
    selBox.style.display = 'none';
});

function clearMultiSelection() {
    document.querySelectorAll('.node.selected').forEach(el => el.classList.remove('selected'));
}

// --- 5. 刪除邏輯 ---
let contextTargetId = null;

function showContextMenu(e, nodeId) {
    contextTargetId = nodeId;
    contextMenu.style.display = 'block';
    contextMenu.style.left = e.clientX + 'px';
    contextMenu.style.top = e.clientY + 'px';
}

document.addEventListener('click', (e) => {
    if(e.target !== contextMenu) contextMenu.style.display = 'none';
});

document.getElementById('btn-delete-node').addEventListener('click', () => {
    if (contextTargetId) deleteNodeAndConnections(contextTargetId);
    contextTargetId = null;
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Delete' || e.key === 'Backspace') {
        const selectedNodes = document.querySelectorAll('.node.selected, .node.active-reactant');
        selectedNodes.forEach(el => deleteNodeAndConnections(el.id));
    }
});

function deleteNodeAndConnections(nodeId) {
    if (!nodes[nodeId]) return;
    board.removeChild(nodes[nodeId].element);
    delete nodes[nodeId];
    
    if (activeReactionNodeId === nodeId) activeReactionNodeId = null;

    connections = connections.filter(c => c.from !== nodeId && c.to !== nodeId);
    drawConnections();
}

// --- 6. 繪圖與 API 邏輯 ---
function drawConnections() {
    svgLayer.innerHTML = '';
    connections.forEach(conn => {
        const nodeA = nodes[conn.from];
        const nodeB = nodes[conn.to];
        if (!nodeA || !nodeB) return;

        const x1 = nodeA.x + nodeA.element.offsetWidth / 2;
        const y1 = nodeA.y + nodeA.element.offsetHeight / 2;
        const x2 = nodeB.x + nodeB.element.offsetWidth / 2;
        const y2 = nodeB.y + nodeB.element.offsetHeight / 2;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1); line.setAttribute('y1', y1);
        line.setAttribute('x2', x2); line.setAttribute('y2', y2);
        line.setAttribute('stroke', conn.isProduct ? '#e67e22' : '#95a5a6');
        line.setAttribute('stroke-width', conn.isProduct ? '3' : '2');
        if (conn.isProduct) line.setAttribute('stroke-dasharray', '5,5');

        svgLayer.appendChild(line);
    });
}

function addConnection(id1, id2, isProduct = false) {
    connections.push({ from: id1, to: id2, isProduct: isProduct });
    drawConnections();
}

async function triggerReaction(nodeA, nodeB) {
    addConnection(nodeA.id, nodeB.id, false);
    
    try {
        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chemicals: [nodeA.value, nodeB.value] })
        });
        const data = await response.json();
        
        if (data.success) {
            const centerX = (nodeA.x + nodeB.x) / 2;
            const baseY = Math.max(nodeA.y, nodeB.y) + 120;
            
            if (data.products && data.products.length > 0) {
                data.products.forEach((prod, index) => {
                    const offsetX = (index * 140) - ((data.products.length - 1) * 70);
                    const productId = createNode(prod.value, prod.name, 'product', centerX + offsetX, baseY);
                    addConnection(nodeA.id, productId, true);
                    addConnection(nodeB.id, productId, true);
                });
            }

            statusBar.children[0].textContent = `✅ 反應成功：${data.equation}`;
        } else {
            statusBar.children[0].textContent = `❌ 無反應：${nodeA.value} 與 ${nodeB.value} 不發生化學變化。`;
        }
    } catch(e) { console.error(e); }
}

window.clearBoard = function() {
    Object.values(nodes).forEach(n => board.removeChild(n.element));
    nodes = {}; connections = []; nodeCounter = 0; activeReactionNodeId = null;
    svgLayer.innerHTML = '';
    statusBar.children[0].textContent = "💡 提示：從左側點擊物質加入，單擊第一個節點，再點擊第二個進行反應。";
}