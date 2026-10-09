import { Component, signal, ViewChild, ElementRef } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';

@Component({
  selector: 'app-root',
  imports: [MatButtonModule],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  @ViewChild('canvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  private get ctx(): CanvasRenderingContext2D {
    return this.canvasRef.nativeElement.getContext('2d')!;
  }
  private drawing = false;

  prediction = signal("-");

  async predict() {
    const canvas = this.canvasRef.nativeElement;
    const base64 = canvas.toDataURL('image/png');
    console.log(base64);

    try {
      const response = await fetch('http://localhost:8080/get-postcode', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          postalImage: base64
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }

      const result = await response.text();
      this.prediction.set(result);
    } catch (error) {
      this.prediction.set("Server Error");
    }
  }

  // === canvas ===
  ngAfterViewInit(): void {
    this.resetCanvas();
  }

  resetCanvas() {
    this.prediction.set("-");
    const canvas = this.canvasRef.nativeElement;
    this.ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw 5 postal code boxes
    const boxWidth = 60;
    const boxHeight = 80;
    const gap = 8;

    const totalWidth = boxWidth * 5 + gap * 4;
    const startX = (canvas.width - totalWidth) / 2;
    const startY = (canvas.height - boxHeight) / 2;

    this.ctx.strokeStyle = '#000000';
    this.ctx.lineWidth = 2;

    for (let i = 0; i < 5; i++) {
      const x = startX + i * (boxWidth + gap);

      this.ctx.strokeRect(x, startY, boxWidth, boxHeight);
    }
  }

  start(event: PointerEvent): void {
    const canvas = this.canvasRef.nativeElement;
    canvas.setPointerCapture(event.pointerId);

    const { x, y } = this.getPosition(event);

    this.drawing = true;
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
  }

  draw(event: PointerEvent): void {
    if (!this.drawing) return;

    const { x, y } = this.getPosition(event);

    this.ctx.lineWidth = 8;
    this.ctx.lineCap = 'round';
    this.ctx.strokeStyle = '#000000';

    this.ctx.lineTo(x, y);
    this.ctx.stroke();
  }

  stop(): void {
    this.drawing = false;
    this.ctx.closePath();
  }

  private getPosition(event: PointerEvent) {
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();

    return {
      x: (event.clientX - rect.left) *
         (this.canvasRef.nativeElement.width / rect.width),
      y: (event.clientY - rect.top) *
         (this.canvasRef.nativeElement.height / rect.height)
    };
  }
}
