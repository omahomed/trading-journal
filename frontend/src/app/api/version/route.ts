export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET() {
  return new Response(
    JSON.stringify({ buildId: process.env.NEXT_PUBLIC_BUILD_ID || "unknown" }),
    {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store, max-age=0, must-revalidate",
      },
    }
  );
}
