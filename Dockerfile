FROM ghcr.io/getzola/zola:v0.19.1 as zola

COPY . /myBlog
WORKDIR /myBlog
RUN ["zola", "build"]

FROM ghcr.io/static-web-server/static-web-server:2
WORKDIR /
COPY --from=zola /myBlog/public /public

# Set environment variables for the static web server
ENV SERVER_ROOT /public
ENV LISTEN_ADDRESS 0.0.0.0
ENV LISTEN_PORT 8080

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Run the static web server
CMD ["static-web-server", "--root", "/public", "--host", "0.0.0.0", "--port", "8080"]

