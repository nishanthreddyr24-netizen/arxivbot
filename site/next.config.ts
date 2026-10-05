import type { NextConfig } from "next";

const API = process.env.API_ORIGIN ?? "http://127.0.0.1:8777";

const nextConfig: NextConfig = {
  /**
   * In development the Python service runs separately, so /api/* is proxied to
   * it and the frontend can use same-origin paths everywhere. On Vercel both
   * halves deploy to one domain and vercel.json routes /api/* to the function,
   * so this rewrite is skipped.
   */
  async rewrites() {
    if (process.env.VERCEL) return [];
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
