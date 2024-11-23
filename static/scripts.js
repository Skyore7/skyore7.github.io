const draggables = document.querySelectorAll('.draggable');
const widthBoard = 1024.0;
const heightBoard = 819.2;
const bordersBoard = 115.2;
const goalDepth = 88.0;

draggables.forEach(item => {
    item.addEventListener('mousedown', onMouseDown);

    function onMouseDown(e) {
        const board = document.getElementById('board');
        const shiftX = e.clientX - item.getBoundingClientRect().left;
        const shiftY = e.clientY - item.getBoundingClientRect().top;

        item.style.position = 'absolute';
        item.style.zIndex = 1000;

        function moveAt(pageX, pageY) {
            const rect = board.getBoundingClientRect();
            var x = Math.min(rect.width - item.offsetWidth, Math.max(0, pageX - rect.left - shiftX));
            var y = Math.min(rect.height - item.offsetHeight, Math.max(0, pageY - rect.top - shiftY));

             // Check if the position is inside the octagon
             //const isInsideOctagon = checkInsideOctagon(x + shiftX, y + shiftY, rect.width, rect.height);
             

             const bottomEdge = heightBoard - bordersBoard;
             const rightEdge = widthBoard - bordersBoard;

            // Top left
            if (x < bordersBoard && y < bordersBoard - x) {
                const xNew = bordersBoard - y;
                const yNew = bordersBoard - x;
                x = (xNew + x) * 0.5
                y = (yNew + y) * 0.5
            } else if (x < bordersBoard && y - bottomEdge + item.offsetWidth > x) {
                const xNew = y - bottomEdge + item.offsetWidth;
                const yNew = x + bottomEdge - item.offsetWidth;
                x = (xNew + x) * 0.5
                y = (yNew + y) * 0.5
            }

             item.style.left = x + 'px';
             item.style.top = y + 'px';

             /*if (isInsideOctagon) {
                 
             }*/
        }

        moveAt(e.pageX, e.pageY);

        function onMouseMove(event) {
            moveAt(event.pageX, event.pageY);
        }

        function onMouseUp() {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        }

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp); // Attach to document
    }

    item.ondragstart = () => false;
});

