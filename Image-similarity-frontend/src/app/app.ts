import { Component, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  selectedFile: File | null = null;
  previewUrl = signal<string | null>(null);

  result = signal<any>(null);
  isLoading = signal(false);
  errorMessage = signal<string | null>(null);

  // Expanded image URL for double-click fullscreen view
  expandedImageUrl = signal<string | null>(null);

  visibleCount = signal(1);

  readonly IMAGE_BASE_URL = 'http://127.0.0.1:8000/images/';

  constructor(private http: HttpClient) {}

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.result.set(null);
      this.errorMessage.set(null);
      this.visibleCount.set(1);

      this.previewUrl.set(URL.createObjectURL(this.selectedFile));
    }
  }

  onSubmit(): void {
    if (!this.selectedFile) {
      this.errorMessage.set('Please select an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    this.isLoading.set(true);
    this.errorMessage.set(null);
    this.visibleCount.set(1);

    this.http.post('http://127.0.0.1:8000/match', formData).subscribe({
      next: (response) => {
        this.result.set(response);
        this.isLoading.set(false);
      },
      error: (err) => {
        this.errorMessage.set('Something went wrong: ' + (err.error?.detail || err.message));
        this.isLoading.set(false);
      }
    });
  }

  showNext(): void {
    const total = this.result()?.top_matches?.length ?? 0;
    if (this.visibleCount() < total) {
      this.visibleCount.update(count => count + 1);
    }
  }

  openLightbox(url: string | null): void {
    if (url) {
      this.expandedImageUrl.set(url);
    }
  }

  closeLightbox(): void {
    this.expandedImageUrl.set(null);
  }
}